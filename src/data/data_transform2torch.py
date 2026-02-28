import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import ast
import os

class RLSequenceDataset(Dataset):
    def __init__(self, dataframe, context_length=20):
        self.data = dataframe.reset_index(drop=True)
        self.context_length = context_length
        
        # Detectar la dimensión del vector de estado automáticamente
        self.state_dim = len(self.data.iloc[0]['observations'][0])
        print(f"Dimensión del Estado (state_dim) detectada: {self.state_dim}")
        
        # Detectar cantidad de acciones posibles
        all_actions = [a for sublist in self.data['actions'] for a in sublist]
        self.act_dim = max(all_actions) + 1
        print(f"Dimensión de Acciones (act_dim) detectada: {self.act_dim}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        states = row['observations']
        actions = row['actions']
        rewards = row['rewards']
        
        # 1. Calcular Return-to-Go (Suma acumulada de recompensas futuras)
        # Esencial para el Decision Transformer
        rtg = np.cumsum(rewards[::-1])[::-1].tolist()
        
        seq_len = len(states)
        
        # 2. Inicializar tensores vacíos (con padding a la IZQUIERDA)
        # Llenamos de ceros inicialmente
        s_tensor = torch.zeros((self.context_length, self.state_dim), dtype=torch.float32)
        a_tensor = torch.zeros((self.context_length, 1), dtype=torch.long)
        rtg_tensor = torch.zeros((self.context_length, 1), dtype=torch.float32)
        mask_tensor = torch.zeros((self.context_length, 1), dtype=torch.float32)
        
        # 3. Truncar si es muy largo
        if seq_len >= self.context_length:
            s_real = states[-self.context_length:]
            a_real = actions[-self.context_length:]
            rtg_real = rtg[-self.context_length:]
            real_len = self.context_length
        else:
            s_real = states
            a_real = actions
            rtg_real = rtg
            real_len = seq_len
            
        # 4. Insertar los datos reales al FINAL del tensor (Padding Izquierdo)
        # Esto ayuda al Transformer a que la acción más reciente esté siempre en la última posición
        s_tensor[-real_len:] = torch.tensor(s_real, dtype=torch.float32)
        a_tensor[-real_len:] = torch.tensor(a_real, dtype=torch.long).unsqueeze(1)
        rtg_tensor[-real_len:] = torch.tensor(rtg_real, dtype=torch.float32).unsqueeze(1)
        mask_tensor[-real_len:] = 1.0  # 1 indica "dato real", 0 indica "padding"
        
        return {
            'states': s_tensor,
            'actions': a_tensor,
            'rtgs': rtg_tensor,
            'attention_mask': mask_tensor
        }

# --- PRUEBA ---
# Asegúrate de pasar tu dataframe real aquí (ej. df_final o trajectories)
# Asumiendo que tu dataframe se llama "trajectories"
trajectories_df = pd.read_csv("/Users/pedro/Desktop/Transformers4Rec/data/processed/trajectorys_df.csv")

print("--- Revisando tipos de datos antes de PyTorch ---")

# 1. Reparar listas convertidas en texto (El problema del CSV)
columnas_lista = ['observations', 'actions', 'rewards']
for col in columnas_lista:
    if col in trajectories_df.columns:
        # Si el primer elemento es un string, significa que Pandas lo leyó como texto
        if isinstance(trajectories_df[col].iloc[0], str):
            print(f"Corrigiendo formato de lista en columna: {col}...")
            trajectories_df[col] = trajectories_df[col].apply(ast.literal_eval)

# 2. Reparar acciones si siguen siendo texto (ej. 'CLICK', 'VIEW')
# Sacamos la primera acción del primer usuario para ver qué tipo de dato es
primera_accion = trajectories_df['actions'].iloc[0][0]

if isinstance(primera_accion, str):
    print("\nDetectadas acciones en formato texto. Convirtiendo a números (Label Encoding)...")
    
    # Encontramos todas las acciones únicas en todo el dataset
    todas_las_acciones = set([a for sublist in trajectories_df['actions'] for a in sublist])
    
    # Creamos un diccionario para mapear: ej. {'CLICK': 0, 'VIEW': 1}
    accion_a_id = {accion: idx for idx, accion in enumerate(todas_las_acciones)}
    print(f"Mapeo de acciones creado: {accion_a_id}")
    
    # Aplicamos la conversión a todas las listas
    trajectories_df['actions'] = trajectories_df['actions'].apply(
        lambda secuencia: [accion_a_id[a] for a in secuencia]
    )

print("\n¡Datos reparados! Creando Dataset de PyTorch...")

dataset = RLSequenceDataset(trajectories_df, context_length=20)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Verificamos que un batch salga con las dimensiones correctas
batch = next(iter(dataloader))
print("\nVerificación del Batch:")
print(f"States shape: {batch['states'].shape} -> Esperado: [64, 20, {dataset.state_dim}]")
print(f"Actions shape: {batch['actions'].shape} -> Esperado: [64, 20, 1]")
print(f"Mask shape: {batch['attention_mask'].shape} -> Esperado: [64, 20, 1]")

import os

print("\n📦 Empaquetando todos los tensores para guardarlos...")

# Extraemos todos los datos y los apilamos en tensores gigantes
all_states = torch.stack([dataset[i]['states'] for i in range(len(dataset))])
all_actions = torch.stack([dataset[i]['actions'] for i in range(len(dataset))])
all_rtgs = torch.stack([dataset[i]['rtgs'] for i in range(len(dataset))])
all_masks = torch.stack([dataset[i]['attention_mask'] for i in range(len(dataset))])

# Creamos un diccionario con los tensores y las dimensiones clave
tensor_data = {
    'states': all_states,
    'actions': all_actions,
    'rtgs': all_rtgs,
    'attention_mask': all_masks,
    'state_dim': dataset.state_dim,
    'act_dim': dataset.act_dim
}

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'tensor_dataset.pt')

# Guardamos el archivo
torch.save(tensor_data, save_path)
print(f"✅ ¡Dataset de tensores guardado exitosamente en:\n{save_path}")