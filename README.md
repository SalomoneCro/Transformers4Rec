# Transformers4Rec

Repositorio de tesis para recomendación secuencial de accionables usando aprendizaje supervisado y RL offline con Transformers.

## Estado del proyecto

Modelos implementados:
- `Popularity` (baseline global)
- `Behavior Cloning (MLP)` en `src/models/behavior_cloning_train.py`
- `Trajectory Transformer` en `src/models/trajectory_transformer_train.py`
- `Decision Transformer` en `src/models/decision_transformer_train.py`
- `Grid Search` para `TT/DT` en `src/models/grid_search_tt_dt.py`

Pipeline reproducible:
- Split temporal persistido (`train/val/test`)
- Artefactos de preprocesamiento persistidos (mapeo de acciones, scaler, columnas de estado)
- Métricas consistentes entre modelos: `HR@K`, `NDCG@K`, `MRR`

## Estructura relevante

```text
src/
  data/
    data_transform2torch.py
    generate_temporal_splits.py
    save_preprocessing_artifacts.py
  models/
    behavior_cloning_train.py
    trajectory_transformer_train.py
    decision_transformer_train.py
    grid_search_tt_dt.py
data/
  raw/
  processed/
models/
papers/
```

## Requisitos

Instalar dependencias (sugerido en entorno virtual `Tenv` ya existente):

```bash
pip install -r requirements.txt
```

## Flujo recomendado

### 1) (Opcional) Regenerar dataset tensorizado

```bash
python src/data/data_transform2torch.py
```

Genera `data/processed/tensor_dataset.pt`.

### 2) Generar artefactos de preprocesamiento

```bash
python src/data/save_preprocessing_artifacts.py
```

Genera `data/processed/preprocessing_artifacts.json` con:
- `action_to_id` / `id_to_action`
- parámetros de `StandardScaler`
- columnas finales del estado

### 3) Generar split temporal persistido + tensores train/val/test

```bash
python src/data/generate_temporal_splits.py
```

Este script:
- hace split temporal usando `trajectorys_df.csv`
- guarda:
  - `data/processed/splits_temporal.json`
  - `data/processed/tensor_dataset_train.pt`
  - `data/processed/tensor_dataset_val.pt`
  - `data/processed/tensor_dataset_test.pt`

### 4) Entrenar baseline BC

```bash
python src/models/behavior_cloning_train.py
```

- Usa los splits persistidos ya generados
- Entrena BC y guarda `models/modelo_baseline_bc.pth`
- Reporta métricas en test para BC y Popularity

### 5) Entrenar Trajectory Transformer

```bash
python src/models/trajectory_transformer_train.py
```

- Usa los mismos splits persistidos
- Incluye early stopping configurable en panel `CONFIG`
- Guarda `models/modelo_trajectory_transformer.pth`
- Reporta métricas test para TT y Popularity

### 6) Entrenar Decision Transformer

```bash
python src/models/decision_transformer_train.py
```

- Usa los mismos splits persistidos y `rtgs`
- Incluye early stopping configurable en panel `CONFIG`
- Guarda `models/modelo_decision_transformer.pth`
- Reporta métricas test para DT y Popularity

## Grid Search TT/DT

```bash
python src/models/grid_search_tt_dt.py
```

Características:
- Selección de modelo(s): `tt`, `dt` o ambos
- Espacio numérico: `min`, `max`, `steps`, `dtype`
- Espacio categórico: `values`
- Progreso impreso por trial (`% modelo` y `% total`)
- Filtrado automático de combinaciones inválidas (por ejemplo `d_model % num_heads != 0`)
- Early stopping por trial

Salidas:
- `data/processed/grid_search/grid_search_results_<timestamp>.csv`
- `data/processed/grid_search/grid_search_best_<timestamp>.json`

## Métricas

Todas las evaluaciones usan:
- `HR@K`
- `NDCG@K`
- `MRR`

Con el mismo esquema de datos y split para asegurar comparabilidad entre BC, TT y DT.

## Configuración

Todos los scripts principales usan panel `CONFIG` dentro del archivo (sin CLI obligatorio):
- épocas
- learning rate
- arquitectura
- early stopping
- top-k
- paths

## Nota de versionado

Por defecto, artefactos grandes generados (`.pth`, splits `.pt`, resultados de grid search) están ignorados por `.gitignore` para mantener el repositorio liviano y enfocado en código/documentación.
