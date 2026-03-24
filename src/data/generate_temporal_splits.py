import ast
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

# =========================
# PANEL DE CONTROL
# =========================
CONFIG = {
    "train_ratio": 0.7,
    "val_ratio": 0.15,
}


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def _resolve_paths() -> Tuple[str, str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    tensor_path = os.path.join(project_root, "data", "processed", "tensor_dataset.pt")
    trajectory_path = os.path.join(project_root, "data", "processed", "trajectorys_df.csv")
    processed_dir = os.path.join(project_root, "data", "processed")
    return tensor_path, trajectory_path, processed_dir


def _load_tensor_dataset(path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe el dataset tensorizado: {path}. "
            "Primero ejecuta src/data/data_transform2torch.py."
        )
    return torch.load(path)


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


def main() -> None:
    train_ratio = float(CONFIG["train_ratio"])
    val_ratio = float(CONFIG["val_ratio"])

    tensor_path, trajectory_path, processed_dir = _resolve_paths()
    print(f"📂 Cargando tensores desde: {tensor_path}")
    print(f"📂 Cargando trayectorias desde: {trajectory_path}")
    tensor_data = _load_tensor_dataset(tensor_path)

    split = temporal_split_indices(
        trajectory_path=trajectory_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    save_split_artifacts(
        processed_dir=processed_dir,
        split=split,
        tensor_data=tensor_data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    print(
        "Split temporal -> "
        f"train={len(split.train)} | val={len(split.val)} | test={len(split.test)}"
    )
    print(
        "Dimensiones -> "
        f"state_dim={int(tensor_data['state_dim'])}, act_dim={int(tensor_data['act_dim'])}"
    )


if __name__ == "__main__":
    main()
