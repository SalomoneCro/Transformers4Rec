import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================
# PANEL DE CONTROL
# =========================
CONFIG = {
    "raw_csv_path": "/Users/pedro/Desktop/Transformers4Rec/data/raw/202512_data_train_SC_GMV.csv",
    "output_json_path": "/Users/pedro/Desktop/Transformers4Rec/data/processed/preprocessing_artifacts.json",
    "id_col": "CUS_CUST_ID_SEL",
    "time_col": "SELLER_TIMESTAMP",
    "action_col": "CARD_KEY_CLICK",
    "reward_col": "REWARD",
    "numeric_cols_heavy": ["ORDERS", "GMV_USD", "VISITS_30D"],
    "numeric_cols_ratio": ["CVR_30D"],
    "categorical_cols": ["SIT_SITE_ID", "SELLER_SEGMENT", "REP_CURRENT_LEVEL"],
    "min_sequence_length": 2,
}


def build_artifacts(config: Dict) -> Dict:
    id_col = config["id_col"]
    time_col = config["time_col"]
    action_col = config["action_col"]
    reward_col = config["reward_col"]
    heavy_cols: List[str] = config["numeric_cols_heavy"]
    ratio_cols: List[str] = config["numeric_cols_ratio"]
    cat_cols: List[str] = config["categorical_cols"]
    min_sequence_length = int(config["min_sequence_length"])

    state_features_base = cat_cols + heavy_cols + ratio_cols
    subset_cols = [id_col, time_col, action_col, reward_col] + state_features_base

    print(f"📂 Cargando raw dataset: {config['raw_csv_path']}")
    df = pd.read_csv(config["raw_csv_path"], low_memory=False)
    df_clean = df[subset_cols].copy()
    df_clean = df_clean.sort_values(by=[id_col, time_col], kind="mergesort")

    all_num_cols = heavy_cols + ratio_cols
    df_clean[all_num_cols] = df_clean[all_num_cols].fillna(0)
    df_clean[cat_cols] = df_clean[cat_cols].fillna("UNKNOWN")

    # Construcción de trayectorias para filtrar usuarios con historial.
    trajectories = (
        df_clean.groupby(id_col)
        .agg({time_col: list, action_col: list, reward_col: list})
        .reset_index()
    )
    seq_len = trajectories[reward_col].apply(len)
    valid_user_ids = set(
        trajectories.loc[seq_len >= min_sequence_length, id_col].tolist()
    )
    df_final = df_clean[df_clean[id_col].isin(valid_user_ids)].copy()

    # Action mapping.
    le_action = LabelEncoder()
    le_action.fit(df_final[action_col].astype(str))
    action_to_id = {
        cls: int(idx) for idx, cls in enumerate(le_action.classes_.tolist())
    }
    id_to_action = {int(v): k for k, v in action_to_id.items()}

    # Numeric preprocessing.
    for col in heavy_cols:
        df_final[col] = np.log1p(df_final[col])
    scaler_cols = heavy_cols + ratio_cols
    scaler = StandardScaler()
    scaler.fit(df_final[scaler_cols])

    # Categorical preprocessing columns.
    df_dummies = pd.get_dummies(
        df_final[cat_cols], columns=cat_cols, prefix=cat_cols, dtype=float, drop_first=False
    )
    categorical_feature_columns = df_dummies.columns.tolist()
    state_feature_columns = scaler_cols + categorical_feature_columns

    artifacts = {
        "source_raw_csv": config["raw_csv_path"],
        "preprocessing_spec": {
            "id_col": id_col,
            "time_col": time_col,
            "action_col": action_col,
            "reward_col": reward_col,
            "min_sequence_length": min_sequence_length,
            "numeric_cols_heavy_log1p_then_scale": heavy_cols,
            "numeric_cols_ratio_scale": ratio_cols,
            "categorical_cols_get_dummies": cat_cols,
        },
        "dataset_stats": {
            "num_rows_raw": int(len(df)),
            "num_rows_after_filter_valid_users": int(len(df_final)),
            "num_users_total": int(trajectories.shape[0]),
            "num_users_valid": int(len(valid_user_ids)),
            "retention_pct_users": float(100.0 * len(valid_user_ids) / max(1, trajectories.shape[0])),
        },
        "action_mapping": {
            "num_actions": int(len(action_to_id)),
            "action_to_id": action_to_id,
            "id_to_action": id_to_action,
        },
        "scaler": {
            "type": "StandardScaler",
            "columns_in_order": scaler_cols,
            "mean_": [float(x) for x in scaler.mean_.tolist()],
            "scale_": [float(x) for x in scaler.scale_.tolist()],
            "var_": [float(x) for x in scaler.var_.tolist()],
            "n_features_in_": int(scaler.n_features_in_),
        },
        "state_columns": {
            "numeric_columns_in_order": scaler_cols,
            "categorical_dummy_columns_in_order": categorical_feature_columns,
            "state_feature_columns_in_order": state_feature_columns,
            "state_dim": int(len(state_feature_columns)),
        },
    }
    return artifacts


def main() -> None:
    artifacts = build_artifacts(CONFIG)
    out_path = CONFIG["output_json_path"]
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, ensure_ascii=True, indent=2)
    print(f"✅ Artefactos guardados en: {out_path}")


if __name__ == "__main__":
    main()
