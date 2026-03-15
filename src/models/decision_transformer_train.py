import os
import random
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
    "seed": 42,
    "epochs": 30,
    "batch_size": 128,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "d_model": 128,
    "num_layers": 3,
    "num_heads": 4,
    "dropout": 0.1,
    "max_context_len": 20,
    "topk": [5, 10],
    "grad_clip_norm": 1.0,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 1e-4,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_paths() -> Tuple[str, str, str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    processed_dir = os.path.join(project_root, "data", "processed")
    train_path = os.path.join(processed_dir, "tensor_dataset_train.pt")
    val_path = os.path.join(processed_dir, "tensor_dataset_val.pt")
    test_path = os.path.join(processed_dir, "tensor_dataset_test.pt")
    model_path = os.path.join(project_root, "models", "modelo_decision_transformer.pth")
    return train_path, val_path, test_path, model_path


def _load_split_tensor(path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe el split tensorizado: {path}. "
            "Primero ejecuta behavior_cloning_train.py para generar los splits persistidos."
        )
    data = torch.load(path)
    if "rtgs" not in data:
        raise KeyError(
            f"El archivo {path} no contiene 'rtgs'. "
            "Regenera los splits con behavior_cloning_train.py actualizado."
        )
    return data


def _build_loader(split_data: Dict[str, torch.Tensor], batch_size: int, shuffle: bool) -> DataLoader:
    states = split_data["states"].float()
    actions = split_data["actions"].long()
    rtgs = split_data["rtgs"].float()
    masks = split_data["attention_mask"].float()
    dataset = TensorDataset(states, actions, rtgs, masks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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

        prev_actions = torch.full(
            (batch_size, seq_len),
            fill_value=self.start_action_id,
            dtype=torch.long,
            device=device,
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
        x = x * padding_mask

        causal_mask = self._causal_mask(seq_len, device=device)
        h = self.encoder(x, mask=causal_mask)
        logits = self.out(h)
        return logits


def _flatten_valid_positions(
    logits: torch.Tensor,
    actions: torch.Tensor,
    masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = actions.view(-1).long()
    valid_idx = masks.view(-1) > 0
    return logits_flat[valid_idx], targets_flat[valid_idx]


def train_one_epoch(
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
    skipped_non_finite = 0

    for states, actions, rtgs, masks in loader:
        states = torch.nan_to_num(states.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        actions = actions.squeeze(-1).to(device)
        rtgs = torch.nan_to_num(rtgs.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(states, actions, rtgs, masks)
        logits_valid, targets_valid = _flatten_valid_positions(logits, actions, masks)
        if targets_valid.numel() == 0:
            continue

        loss = criterion(logits_valid, targets_valid)
        if not torch.isfinite(loss):
            skipped_non_finite += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        pred = logits_valid.argmax(dim=1)
        total_correct += (pred == targets_valid).sum().item()
        total_count += targets_valid.numel()

    if skipped_non_finite > 0:
        print(f"⚠️ Batches omitidos por loss no finita: {skipped_non_finite}")
    return total_loss / max(1, len(loader)), (100.0 * total_correct / max(1, total_count))


@torch.no_grad()
def collect_eval_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for states, actions, rtgs, masks in loader:
        states = torch.nan_to_num(states.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        actions = actions.squeeze(-1).to(device)
        rtgs = torch.nan_to_num(rtgs.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        masks = masks.to(device)

        logits = model(states, actions, rtgs, masks)
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
    logits: torch.Tensor,
    targets: torch.Tensor,
    ks: Sequence[int],
) -> Dict[str, float]:
    if logits.numel() == 0 or targets.numel() == 0:
        metrics = {f"HR@{k}": 0.0 for k in ks}
        metrics.update({f"NDCG@{k}": 0.0 for k in ks})
        metrics["MRR"] = 0.0
        return metrics

    ranking = torch.argsort(logits, dim=1, descending=True)
    target_np = targets.numpy()
    ranks = []
    for i, tgt in enumerate(target_np):
        row = ranking[i].numpy()
        match = np.where(row == tgt)[0]
        rank = int(match[0]) + 1 if len(match) else len(row) + 1
        ranks.append(rank)
    ranks = np.array(ranks, dtype=np.int64)

    out: Dict[str, float] = {}
    for k in ks:
        hits = (ranks <= k).astype(np.float32)
        ndcgs = np.array([_dcg_gain(r - 1) if r <= k else 0.0 for r in ranks], dtype=np.float32)
        out[f"HR@{k}"] = float(hits.mean())
        out[f"NDCG@{k}"] = float(ndcgs.mean())
    out["MRR"] = float((1.0 / ranks).mean())
    return out


def popularity_ranking_from_train(loader: DataLoader, num_actions: int) -> torch.Tensor:
    counts = torch.zeros(num_actions, dtype=torch.long)
    for _, actions, _, masks in loader:
        actions = actions.squeeze(-1)
        valid = masks.view(-1) > 0
        y = actions.view(-1)[valid]
        if y.numel() == 0:
            continue
        counts += torch.bincount(y, minlength=num_actions)
    return torch.argsort(counts, descending=True)


def topk_metrics_from_fixed_ranking(
    ranking: torch.Tensor,
    targets: torch.Tensor,
    ks: Sequence[int],
) -> Dict[str, float]:
    if targets.numel() == 0:
        metrics = {f"HR@{k}": 0.0 for k in ks}
        metrics.update({f"NDCG@{k}": 0.0 for k in ks})
        metrics["MRR"] = 0.0
        return metrics

    rank_map = torch.empty_like(ranking)
    rank_map[ranking] = torch.arange(1, len(ranking) + 1)
    ranks_np = rank_map[targets].numpy()

    out: Dict[str, float] = {}
    for k in ks:
        hits = (ranks_np <= k).astype(np.float32)
        ndcg_vals = np.array([_dcg_gain(r - 1) if r <= k else 0.0 for r in ranks_np], dtype=np.float32)
        out[f"HR@{k}"] = float(hits.mean())
        out[f"NDCG@{k}"] = float(ndcg_vals.mean())
    out["MRR"] = float((1.0 / ranks_np).mean())
    return out


def format_metrics(title: str, metrics: Dict[str, float], ks: Iterable[int]) -> str:
    ordered = [f"HR@{k}={metrics[f'HR@{k}']:.4f}" for k in ks]
    ordered += [f"NDCG@{k}={metrics[f'NDCG@{k}']:.4f}" for k in ks]
    ordered.append(f"MRR={metrics['MRR']:.4f}")
    return f"{title}: " + " | ".join(ordered)


def main() -> None:
    set_seed(int(CONFIG["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path, val_path, test_path, model_path = _resolve_paths()
    print(f"📂 Cargando splits persistidos:\n- {train_path}\n- {val_path}\n- {test_path}")

    train_data = _load_split_tensor(train_path)
    val_data = _load_split_tensor(val_path)
    test_data = _load_split_tensor(test_path)

    state_dim = int(train_data["state_dim"])
    act_dim = int(train_data["act_dim"])
    seq_len = int(train_data["states"].shape[1])
    print(
        "Dimensiones -> "
        f"state_dim={state_dim}, act_dim={act_dim}, context_len={seq_len}"
    )

    max_context = max(int(CONFIG["max_context_len"]), seq_len)
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        d_model=int(CONFIG["d_model"]),
        num_layers=int(CONFIG["num_layers"]),
        num_heads=int(CONFIG["num_heads"]),
        dropout=float(CONFIG["dropout"]),
        max_context_len=max_context,
    ).to(device)

    batch_size = int(CONFIG["batch_size"])
    topk = [int(k) for k in CONFIG["topk"]]
    train_loader = _build_loader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = _build_loader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = _build_loader(test_data, batch_size=batch_size, shuffle=False)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(CONFIG["lr"]),
        weight_decay=float(CONFIG["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    patience = int(CONFIG["early_stopping_patience"])
    min_delta = float(CONFIG["early_stopping_min_delta"])

    epochs = int(CONFIG["epochs"])
    print(f"🚀 Entrenando Decision Transformer por {epochs} épocas...")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip_norm=float(CONFIG["grad_clip_norm"]),
        )
        val_logits, val_targets = collect_eval_logits(model, val_loader, device)
        if val_targets.numel() == 0:
            val_loss = float("inf")
        elif not torch.isfinite(val_logits).all():
            val_loss = float("inf")
        else:
            val_loss = criterion(val_logits, val_targets).item()

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.2f}% | val_loss={val_loss:.4f}"
        )
        if epochs_without_improvement >= patience:
            print(
                "⏹️ Early stopping activado: "
                f"sin mejora de val_loss por {patience} épocas."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), model_path)
    print(f"💾 Modelo guardado en: {model_path}")

    test_logits, test_targets = collect_eval_logits(model, test_loader, device)
    dt_metrics = topk_metrics_from_logits(test_logits, test_targets, topk)

    popularity_rank = popularity_ranking_from_train(train_loader, num_actions=act_dim)
    pop_metrics = topk_metrics_from_fixed_ranking(popularity_rank, test_targets, topk)

    print("\n=== Evaluación Test ===")
    print(format_metrics("DecisionTransformer", dt_metrics, topk))
    print(format_metrics("Popularity", pop_metrics, topk))


if __name__ == "__main__":
    main()
