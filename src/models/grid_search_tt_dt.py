import csv
import itertools
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# =========================
# PANEL DE CONTROL
# =========================
CONFIG = {
    # "tt", "dt" o ambos: ["tt", "dt"]
    "models_to_run": ["tt", "dt"],
    "seed": 42,
    "device": "auto",  # "auto", "cpu", "cuda"
    "optimization_metric": "val_loss",  # se minimiza
    "save_dir": "/Users/pedro/Desktop/Transformers4Rec/data/processed/grid_search",
    "max_trials_per_model": None,  # None = todos los combos
    "training": {
        "epochs": 30,
        "grad_clip_norm": 1.0,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 1e-4,
    },
    # Espacio de búsqueda: numeric(min/max/steps + dtype) o categorical(values)
    "search_space": {
        "lr": {"type": "numeric", "min": 1e-4, "max": 1e-2, "steps": 5, "dtype": "float"},
        "weight_decay": {"type": "categorical", "values": [1e-3, 1e-4]},
        "d_model": {"type": "categorical", "values": [128]},
        "num_layers": {"type": "categorical", "values": [3, 4]},
        "num_heads": {"type": "categorical", "values": [2, 4]},
        "dropout": {"type": "categorical", "values": [0.1, 0.2]},
        "batch_size": {"type": "categorical", "values": [128]},
        "max_context_len": {"type": "categorical", "values": [10, 20]},
        "topk": {"type": "categorical", "values": [[5, 10]]},
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_paths() -> Tuple[str, str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    processed_dir = os.path.join(project_root, "data", "processed")
    train_path = os.path.join(processed_dir, "tensor_dataset_train.pt")
    val_path = os.path.join(processed_dir, "tensor_dataset_val.pt")
    test_path = os.path.join(processed_dir, "tensor_dataset_test.pt")
    return train_path, val_path, test_path


def _load_split_tensor(path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe split tensorizado: {path}. "
            "Ejecuta behavior_cloning_train.py para generar splits."
        )
    return torch.load(path)


def _build_loader(
    split_data: Dict[str, torch.Tensor], batch_size: int, shuffle: bool, use_rtg: bool
) -> DataLoader:
    states = split_data["states"].float()
    actions = split_data["actions"].long()
    masks = split_data["attention_mask"].float()
    if use_rtg:
        if "rtgs" not in split_data:
            raise KeyError("El split no tiene 'rtgs', requerido para Decision Transformer.")
        rtgs = split_data["rtgs"].float()
        dataset = TensorDataset(states, actions, rtgs, masks)
    else:
        dataset = TensorDataset(states, actions, masks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class TrajectoryTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_context_len: int,
    ) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.start_action_id = act_dim
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Embedding(act_dim + 1, d_model)
        self.pos_embed = nn.Embedding(max_context_len, d_model)
        self.input_ln = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, act_dim)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = states.shape
        device = states.device
        valid_mask = padding_mask
        key_padding_mask = ~valid_mask.squeeze(-1).bool()
        prev_actions = torch.full(
            (batch_size, seq_len), fill_value=self.start_action_id, dtype=torch.long, device=device
        )
        prev_actions[:, 1:] = actions[:, :-1]
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.state_proj(states) + self.action_embed(prev_actions) + self.pos_embed(pos_ids)
        x = self.input_ln(x)
        x = x * valid_mask
        # Con padding a la derecha, las posiciones padded quedan fuera del tramo visible
        # para los tokens reales cuando se combina la causal mask con la key padding mask.
        h = self.encoder(
            x,
            mask=self._causal_mask(seq_len, device),
            src_key_padding_mask=key_padding_mask,
        )
        h = h * valid_mask
        return self.out(h)


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_context_len: int,
    ) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.start_action_id = act_dim
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Embedding(act_dim + 1, d_model)
        self.rtg_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Embedding(max_context_len, d_model)
        self.input_ln = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, act_dim)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = states.shape
        device = states.device
        valid_mask = padding_mask
        key_padding_mask = ~valid_mask.squeeze(-1).bool()
        prev_actions = torch.full(
            (batch_size, seq_len), fill_value=self.start_action_id, dtype=torch.long, device=device
        )
        prev_actions[:, 1:] = actions[:, :-1]
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = (
            self.state_proj(states)
            + self.action_embed(prev_actions)
            + self.rtg_proj(rtgs)
            + self.pos_embed(pos_ids)
        )
        x = self.input_ln(x)
        x = x * valid_mask
        # Con padding a la derecha, las posiciones padded quedan fuera del tramo visible
        # para los tokens reales cuando se combina la causal mask con la key padding mask.
        h = self.encoder(
            x,
            mask=self._causal_mask(seq_len, device),
            src_key_padding_mask=key_padding_mask,
        )
        h = h * valid_mask
        return self.out(h)


def _flatten_valid_positions(
    logits: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = actions.view(-1).long()
    valid = masks.view(-1) > 0
    return logits_flat[valid], targets_flat[valid]


def train_one_epoch(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        optimizer.zero_grad()
        if model_name == "dt":
            states, actions, rtgs, masks = batch
            states = torch.nan_to_num(states.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            actions = actions.squeeze(-1).to(device)
            rtgs = torch.nan_to_num(rtgs.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            masks = masks.to(device)
            logits = model(states, actions, rtgs, masks)
        else:
            states, actions, masks = batch
            states = torch.nan_to_num(states.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            actions = actions.squeeze(-1).to(device)
            masks = masks.to(device)
            logits = model(states, actions, masks)

        logits_valid, targets_valid = _flatten_valid_positions(logits, actions, masks)
        if targets_valid.numel() == 0:
            continue
        loss = criterion(logits_valid, targets_valid)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        pred = logits_valid.argmax(dim=1)
        total_correct += (pred == targets_valid).sum().item()
        total_count += targets_valid.numel()

    return total_loss / max(1, len(loader)), (100.0 * total_correct / max(1, total_count))


@torch.no_grad()
def collect_eval_logits(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for batch in loader:
        if model_name == "dt":
            states, actions, rtgs, masks = batch
            states = torch.nan_to_num(states.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            actions = actions.squeeze(-1).to(device)
            rtgs = torch.nan_to_num(rtgs.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            masks = masks.to(device)
            logits = model(states, actions, rtgs, masks)
        else:
            states, actions, masks = batch
            states = torch.nan_to_num(states.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            actions = actions.squeeze(-1).to(device)
            masks = masks.to(device)
            logits = model(states, actions, masks)

        logits_valid, targets_valid = _flatten_valid_positions(logits, actions, masks)
        if targets_valid.numel() == 0:
            continue
        all_logits.append(logits_valid.cpu())
        all_targets.append(targets_valid.cpu())

    if not all_logits:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long)
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def _dcg_gain(rank: int) -> float:
    return 1.0 / np.log2(rank + 2.0)


def topk_metrics_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, ks: Sequence[int]
) -> Dict[str, float]:
    if logits.numel() == 0 or targets.numel() == 0:
        m = {f"HR@{k}": 0.0 for k in ks}
        m.update({f"NDCG@{k}": 0.0 for k in ks})
        m["MRR"] = 0.0
        return m

    ranking = torch.argsort(logits, dim=1, descending=True)
    target_np = targets.numpy()
    ranks = []
    for i, tgt in enumerate(target_np):
        row = ranking[i].numpy()
        match = np.where(row == tgt)[0]
        rank = int(match[0]) + 1 if len(match) else len(row) + 1
        ranks.append(rank)
    ranks_np = np.array(ranks, dtype=np.int64)

    out: Dict[str, float] = {}
    for k in ks:
        hits = (ranks_np <= k).astype(np.float32)
        ndcgs = np.array([_dcg_gain(r - 1) if r <= k else 0.0 for r in ranks_np], dtype=np.float32)
        out[f"HR@{k}"] = float(hits.mean())
        out[f"NDCG@{k}"] = float(ndcgs.mean())
    out["MRR"] = float((1.0 / ranks_np).mean())
    return out


def values_from_spec(spec: Dict) -> List:
    stype = spec["type"]
    if stype == "categorical":
        return list(spec["values"])
    if stype == "numeric":
        vmin = spec["min"]
        vmax = spec["max"]
        steps = int(spec["steps"])
        dtype = spec.get("dtype", "float")
        if steps <= 1:
            vals = [vmin]
        else:
            vals = np.linspace(vmin, vmax, num=steps).tolist()
        if dtype == "int":
            vals = [int(round(v)) for v in vals]
            # evita repetidos por rounding
            dedup = []
            for v in vals:
                if v not in dedup:
                    dedup.append(v)
            vals = dedup
        return vals
    raise ValueError(f"Tipo de spec no soportado: {stype}")


def build_grid(search_space: Dict[str, Dict]) -> List[Dict]:
    keys = list(search_space.keys())
    values_lists = [values_from_spec(search_space[k]) for k in keys]
    combos = []
    for values in itertools.product(*values_lists):
        combos.append({k: v for k, v in zip(keys, values)})
    return combos


def is_valid_hyperparam_combo(hp: Dict) -> Tuple[bool, str]:
    d_model = int(hp["d_model"])
    num_heads = int(hp["num_heads"])
    if d_model % num_heads != 0:
        return False, "d_model debe ser divisible por num_heads"
    return True, ""


def create_model(
    model_name: str,
    state_dim: int,
    act_dim: int,
    seq_len: int,
    hp: Dict,
    device: torch.device,
) -> nn.Module:
    max_context = max(int(hp["max_context_len"]), seq_len)
    if model_name == "tt":
        return TrajectoryTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            d_model=int(hp["d_model"]),
            num_layers=int(hp["num_layers"]),
            num_heads=int(hp["num_heads"]),
            dropout=float(hp["dropout"]),
            max_context_len=max_context,
        ).to(device)
    if model_name == "dt":
        return DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            d_model=int(hp["d_model"]),
            num_layers=int(hp["num_layers"]),
            num_heads=int(hp["num_heads"]),
            dropout=float(hp["dropout"]),
            max_context_len=max_context,
        ).to(device)
    raise ValueError(f"Modelo no soportado: {model_name}")


def run_trial(
    model_name: str,
    hp: Dict,
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    device: torch.device,
    training_cfg: Dict,
) -> Dict:
    use_rtg = model_name == "dt"
    batch_size = int(hp["batch_size"])
    topk = list(hp["topk"])
    train_loader = _build_loader(train_data, batch_size=batch_size, shuffle=True, use_rtg=use_rtg)
    val_loader = _build_loader(val_data, batch_size=batch_size, shuffle=False, use_rtg=use_rtg)
    test_loader = _build_loader(test_data, batch_size=batch_size, shuffle=False, use_rtg=use_rtg)

    state_dim = int(train_data["state_dim"])
    act_dim = int(train_data["act_dim"])
    seq_len = int(train_data["states"].shape[1])
    model = create_model(model_name, state_dim, act_dim, seq_len, hp, device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    epochs = int(training_cfg["epochs"])
    patience = int(training_cfg["early_stopping_patience"])
    min_delta = float(training_cfg["early_stopping_min_delta"])
    grad_clip_norm = float(training_cfg["grad_clip_norm"])

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model_name=model_name,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip_norm=grad_clip_norm,
        )
        val_logits, val_targets = collect_eval_logits(model_name, model, val_loader, device)
        if val_targets.numel() == 0 or not torch.isfinite(val_logits).all():
            val_loss = float("inf")
        else:
            val_loss = criterion(val_logits, val_targets).item()

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_logits, test_targets = collect_eval_logits(model_name, model, test_loader, device)
    metrics = topk_metrics_from_logits(test_logits, test_targets, topk)
    elapsed = time.time() - t0

    result = {
        "model": model_name,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "train_loss_last": float(train_loss),
        "train_acc_last": float(train_acc),
        "elapsed_sec": float(elapsed),
        "HR@5": float(metrics.get("HR@5", np.nan)),
        "HR@10": float(metrics.get("HR@10", np.nan)),
        "NDCG@5": float(metrics.get("NDCG@5", np.nan)),
        "NDCG@10": float(metrics.get("NDCG@10", np.nan)),
        "MRR": float(metrics["MRR"]),
        "hyperparams_json": json.dumps(hp, ensure_ascii=True),
    }
    for k, v in hp.items():
        result[f"hp_{k}"] = v
    return result


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_results_csv(results: List[Dict], out_csv: str) -> None:
    if not results:
        return
    fieldnames = sorted(set().union(*(r.keys() for r in results)))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def best_per_model(results: List[Dict], metric: str) -> Dict[str, Dict]:
    by_model: Dict[str, List[Dict]] = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)
    out: Dict[str, Dict] = {}
    for model, rows in by_model.items():
        rows_sorted = sorted(rows, key=lambda x: x[metric])
        out[model] = rows_sorted[0]
    return out


def main() -> None:
    set_seed(int(CONFIG["seed"]))
    device = resolve_device(str(CONFIG["device"]))
    print(f"🖥️ Device: {device}")

    train_path, val_path, test_path = _resolve_paths()
    train_data = _load_split_tensor(train_path)
    val_data = _load_split_tensor(val_path)
    test_data = _load_split_tensor(test_path)

    models_to_run = list(CONFIG["models_to_run"])
    for m in models_to_run:
        if m not in ("tt", "dt"):
            raise ValueError(f"Modelo inválido en models_to_run: {m}")

    grid = build_grid(CONFIG["search_space"])
    if not grid:
        raise ValueError("El search_space no generó combinaciones.")
    print(f"🔎 Total combinaciones por modelo (sin filtrar): {len(grid)}")

    valid_grid: List[Dict] = []
    invalid_grid: List[Tuple[Dict, str]] = []
    for hp in grid:
        is_valid, reason = is_valid_hyperparam_combo(hp)
        if is_valid:
            valid_grid.append(hp)
        else:
            invalid_grid.append((hp, reason))
    grid = valid_grid

    if invalid_grid:
        print(f"⚠️ Combinaciones inválidas descartadas: {len(invalid_grid)}")
        # Muestra un ejemplo para diagnóstico rápido.
        sample_hp, sample_reason = invalid_grid[0]
        print(f"   Ejemplo descartado: hp={sample_hp} -> {sample_reason}")

    print(f"🔎 Total combinaciones por modelo (válidas): {len(grid)}")
    if not grid:
        raise ValueError(
            "No quedaron combinaciones válidas luego del filtrado. "
            "Revisá search_space (ej: d_model divisible por num_heads)."
        )

    max_trials = CONFIG["max_trials_per_model"]
    if isinstance(max_trials, int) and max_trials > 0:
        grid = grid[:max_trials]
        print(f"🔎 Limitando a {len(grid)} trials por modelo (max_trials_per_model).")

    all_results: List[Dict] = []
    total_trials_all_models = len(grid) * len(models_to_run)
    completed_trials_all_models = 0
    for model_name in models_to_run:
        print(f"\n=== Grid Search {model_name.upper()} ===")
        for idx, hp in enumerate(grid, start=1):
            completed_trials_all_models += 1
            pct_model = 100.0 * idx / max(1, len(grid))
            pct_total = 100.0 * completed_trials_all_models / max(1, total_trials_all_models)
            print(
                f"[{model_name}] Trial {idx}/{len(grid)} "
                f"({pct_model:.2f}% modelo | {pct_total:.2f}% total) | hp={hp}"
            )
            result = run_trial(
                model_name=model_name,
                hp=hp,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                device=device,
                training_cfg=CONFIG["training"],
            )
            all_results.append(result)
            print(
                f"  best_val_loss={result['best_val_loss']:.4f} | "
                f"HR@10={result.get('HR@10', np.nan):.4f} | MRR={result['MRR']:.4f}"
            )

    ensure_dir(CONFIG["save_dir"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(CONFIG["save_dir"], f"grid_search_results_{ts}.csv")
    out_best = os.path.join(CONFIG["save_dir"], f"grid_search_best_{ts}.json")
    save_results_csv(all_results, out_csv)

    best = best_per_model(all_results, metric=str(CONFIG["optimization_metric"]))
    payload = {
        "timestamp": ts,
        "optimization_metric": CONFIG["optimization_metric"],
        "models_to_run": models_to_run,
        "num_trials_total": len(all_results),
        "best_by_model": best,
    }
    with open(out_best, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    print("\n✅ Grid search finalizado.")
    print(f"Resultados: {out_csv}")
    print(f"Mejores combinaciones: {out_best}")


if __name__ == "__main__":
    main()
