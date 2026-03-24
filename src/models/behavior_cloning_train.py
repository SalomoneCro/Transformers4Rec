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
    "hidden_dim": 128,
    "lr": 1e-3,
    "topk": [5, 10],
}


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


def _resolve_project_paths() -> Tuple[str, str, str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    processed_dir = os.path.join(project_root, "data", "processed")
    train_path = os.path.join(processed_dir, "tensor_dataset_train.pt")
    val_path = os.path.join(processed_dir, "tensor_dataset_val.pt")
    test_path = os.path.join(processed_dir, "tensor_dataset_test.pt")
    model_path = os.path.join(project_root, "models", "modelo_baseline_bc.pth")
    return train_path, val_path, test_path, model_path


def _load_split_tensor(path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe el split tensorizado: {path}. "
            "Primero ejecuta src/data/generate_temporal_splits.py para preparar train/val/test."
        )
    return torch.load(path)


def _build_loader(
    split_data: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    states = split_data["states"].float()
    actions = split_data["actions"].long()
    masks = split_data["attention_mask"].float()
    dataset = TensorDataset(states, actions, masks)
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

    train_path, val_path, test_path, model_path = _resolve_project_paths()
    print(f"📂 Cargando splits persistidos:\n- {train_path}\n- {val_path}\n- {test_path}")

    train_data = _load_split_tensor(train_path)
    val_data = _load_split_tensor(val_path)
    test_data = _load_split_tensor(test_path)

    input_dim = int(train_data["state_dim"])
    output_dim = int(train_data["act_dim"])
    seq_len = int(train_data["states"].shape[1])
    print(
        "Dimensiones -> "
        f"state_dim={input_dim}, act_dim={output_dim}, context_len={seq_len}"
    )

    batch_size = int(CONFIG["batch_size"])
    topk = [int(k) for k in CONFIG["topk"]]

    train_loader = _build_loader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = _build_loader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = _build_loader(test_data, batch_size=batch_size, shuffle=False)

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
