import ast
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
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
    "hidden_dim": 128,
    "lr": 1e-3,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "topk": [5, 10],
    "save_split_artifacts": True,
}


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


class BehaviorCloningBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_project_paths() -> Tuple[str, str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    tensor_path = os.path.join(project_root, "data", "processed", "tensor_dataset.pt")
    trajectory_path = os.path.join(project_root, "data", "processed", "trajectorys_df.csv")
    model_path = os.path.join(project_root, "models", "modelo_baseline_bc.pth")
    processed_dir = os.path.join(project_root, "data", "processed")
    return tensor_path, trajectory_path, model_path, processed_dir


def save_split_artifacts(
    processed_dir: str,
    split: SplitIndices,
    tensor_data: Dict[str, torch.Tensor],
    train_ratio: float,
    val_ratio: float,
) -> None:
    os.makedirs(processed_dir, exist_ok=True)

    split_json_path = os.path.join(processed_dir, "splits_temporal.json")
    split_payload = {
        "strategy": "temporal_last_timestamp",
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "num_samples": int(len(split.train) + len(split.val) + len(split.test)),
        "train_size": int(len(split.train)),
        "val_size": int(len(split.val)),
        "test_size": int(len(split.test)),
        "indices": {
            "train": split.train.tolist(),
            "val": split.val.tolist(),
            "test": split.test.tolist(),
        },
    }
    with open(split_json_path, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, ensure_ascii=True, indent=2)

    states = tensor_data["states"].float()
    actions = tensor_data["actions"].long()
    masks = tensor_data["attention_mask"].float()
    rtgs = tensor_data["rtgs"].float() if "rtgs" in tensor_data else None
    state_dim = int(tensor_data["state_dim"])
    act_dim = int(tensor_data["act_dim"])

    split_to_idx = {
        "train": torch.tensor(split.train, dtype=torch.long),
        "val": torch.tensor(split.val, dtype=torch.long),
        "test": torch.tensor(split.test, dtype=torch.long),
    }

    for split_name, idx in split_to_idx.items():
        payload = {
            "states": states[idx],
            "actions": actions[idx],
            "attention_mask": masks[idx],
            "state_dim": state_dim,
            "act_dim": act_dim,
        }
        if rtgs is not None:
            payload["rtgs"] = rtgs[idx]
        out_path = os.path.join(processed_dir, f"tensor_dataset_{split_name}.pt")
        torch.save(payload, out_path)

    print(f"🗂️ Split guardado en: {split_json_path}")
    print(
        "🗂️ Tensores por split guardados en: "
        f"{os.path.join(processed_dir, 'tensor_dataset_train.pt')}, "
        f"{os.path.join(processed_dir, 'tensor_dataset_val.pt')}, "
        f"{os.path.join(processed_dir, 'tensor_dataset_test.pt')}"
    )


def _parse_last_timestamp(value: str) -> pd.Timestamp:
    timestamps = ast.literal_eval(value) if isinstance(value, str) else value
    if not timestamps:
        return pd.NaT
    return pd.to_datetime(timestamps[-1], utc=True, errors="coerce")


def temporal_split_indices(
    trajectory_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> SplitIndices:
    df = pd.read_csv(trajectory_path)
    df["_row_id"] = np.arange(len(df))
    df["_last_ts"] = df["timestamps"].apply(_parse_last_timestamp)
    df = df.sort_values("_last_ts", kind="mergesort").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    all_indices = df["_row_id"].to_numpy()
    train_idx = all_indices[:train_end]
    val_idx = all_indices[train_end:val_end]
    test_idx = all_indices[val_end:]
    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


def _build_loader(
    states: torch.Tensor,
    actions: torch.Tensor,
    masks: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    dataset = TensorDataset(states[idx_tensor], actions[idx_tensor], masks[idx_tensor])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _flatten_valid_positions(
    batch_states: torch.Tensor,
    batch_actions: torch.Tensor,
    batch_masks: torch.Tensor,
    input_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    states_flat = batch_states.view(-1, input_dim)
    actions_flat = batch_actions.view(-1).long()
    mask_flat = batch_masks.view(-1)
    valid_idx = mask_flat > 0
    return states_flat[valid_idx], actions_flat[valid_idx]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    input_dim: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_states, batch_actions, batch_masks in loader:
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)
        batch_masks = batch_masks.to(device)

        x, y_true = _flatten_valid_positions(batch_states, batch_actions, batch_masks, input_dim)
        if x.numel() == 0:
            continue

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        y_pred = logits.argmax(dim=1)
        correct += (y_pred == y_true).sum().item()
        total += y_true.numel()

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def collect_eval_logits(
    model: nn.Module,
    loader: DataLoader,
    input_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for batch_states, batch_actions, batch_masks in loader:
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)
        batch_masks = batch_masks.to(device)

        x, y_true = _flatten_valid_positions(batch_states, batch_actions, batch_masks, input_dim)
        if x.numel() == 0:
            continue

        logits = model(x)
        all_logits.append(logits.cpu())
        all_targets.append(y_true.cpu())

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

    probs_rank = torch.argsort(logits, dim=1, descending=True)
    targets_np = targets.numpy()
    ranks = []

    for i, target in enumerate(targets_np):
        order = probs_rank[i].numpy()
        match = np.where(order == target)[0]
        rank = int(match[0]) + 1 if len(match) else len(order) + 1
        ranks.append(rank)

    ranks_np = np.array(ranks, dtype=np.int64)
    out: Dict[str, float] = {}

    for k in ks:
        hits = (ranks_np <= k).astype(np.float32)
        ndcg_vals = np.array([_dcg_gain(r - 1) if r <= k else 0.0 for r in ranks_np], dtype=np.float32)
        out[f"HR@{k}"] = float(hits.mean())
        out[f"NDCG@{k}"] = float(ndcg_vals.mean())

    out["MRR"] = float((1.0 / ranks_np).mean())
    return out


def popularity_ranking_from_train(
    loader: DataLoader,
    input_dim: int,
    num_actions: int,
) -> torch.Tensor:
    counts = torch.zeros(num_actions, dtype=torch.long)
    for batch_states, batch_actions, batch_masks in loader:
        _, y_true = _flatten_valid_positions(batch_states, batch_actions, batch_masks, input_dim)
        if y_true.numel() == 0:
            continue
        counts += torch.bincount(y_true, minlength=num_actions)
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

    tensor_path, trajectory_path, model_path, processed_dir = _resolve_project_paths()
    print(f"📂 Cargando tensores desde: {tensor_path}")
    tensor_data = torch.load(tensor_path)

    states = tensor_data["states"].float()
    actions = tensor_data["actions"].long()
    masks = tensor_data["attention_mask"].float()
    input_dim = int(tensor_data["state_dim"])
    output_dim = int(tensor_data["act_dim"])

    split = temporal_split_indices(
        trajectory_path=trajectory_path,
        train_ratio=float(CONFIG["train_ratio"]),
        val_ratio=float(CONFIG["val_ratio"]),
    )
    if bool(CONFIG["save_split_artifacts"]):
        save_split_artifacts(
            processed_dir=processed_dir,
            split=split,
            tensor_data=tensor_data,
            train_ratio=float(CONFIG["train_ratio"]),
            val_ratio=float(CONFIG["val_ratio"]),
        )

    print(
        "Split temporal -> "
        f"train={len(split.train)} | val={len(split.val)} | test={len(split.test)}"
    )
    print(f"Dimensiones -> state_dim={input_dim}, act_dim={output_dim}")

    batch_size = int(CONFIG["batch_size"])
    topk = [int(k) for k in CONFIG["topk"]]

    train_loader = _build_loader(states, actions, masks, split.train, batch_size, shuffle=True)
    val_loader = _build_loader(states, actions, masks, split.val, batch_size, shuffle=False)
    test_loader = _build_loader(states, actions, masks, split.test, batch_size, shuffle=False)

    model_bc = BehaviorCloningBaseline(input_dim, int(CONFIG["hidden_dim"]), output_dim).to(device)
    optimizer = optim.Adam(model_bc.parameters(), lr=float(CONFIG["lr"]))
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None

    epochs = int(CONFIG["epochs"])
    print(f"🚀 Entrenando BC por {epochs} épocas...")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model_bc, train_loader, optimizer, criterion, input_dim, device
        )

        val_logits, val_targets = collect_eval_logits(model_bc, val_loader, input_dim, device)
        val_loss = criterion(val_logits, val_targets).item() if val_targets.numel() > 0 else float("inf")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model_bc.state_dict().items()}

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.2f}% | val_loss={val_loss:.4f}"
        )

    if best_state is not None:
        model_bc.load_state_dict(best_state)

    torch.save(model_bc.state_dict(), model_path)
    print(f"💾 Modelo BC guardado en: {model_path}")

    test_logits, test_targets = collect_eval_logits(model_bc, test_loader, input_dim, device)
    bc_metrics = topk_metrics_from_logits(test_logits, test_targets, topk)

    popularity_rank = popularity_ranking_from_train(train_loader, input_dim, output_dim)
    pop_metrics = topk_metrics_from_fixed_ranking(popularity_rank, test_targets, topk)

    print("\n=== Evaluación Test ===")
    print(format_metrics("BehaviorCloning", bc_metrics, topk))
    print(format_metrics("Popularity", pop_metrics, topk))


if __name__ == "__main__":
    main()
