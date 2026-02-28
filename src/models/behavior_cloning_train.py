import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# 1. Obtiene la ruta absoluta de la carpeta de este script (src/models/)
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Sube dos niveles hasta la raíz y entra a data/processed/
data_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'processed', 'tensor_dataset.pt'))
print(f"📂 Cargando datos desde: {data_path}")
tensor_data = torch.load(data_path)

states = tensor_data['states']
actions = tensor_data['actions']
masks = tensor_data['attention_mask']
input_dim = tensor_data['state_dim']
output_dim = tensor_data['act_dim']

print(f"Dimensiones cargadas -> Estado: {input_dim}, Acciones: {output_dim}")

# 2. Crear DataLoader nativo de PyTorch
dataset = TensorDataset(states, actions, masks)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 3. Definir la Red Neuronal (MLP)
class BehaviorCloningBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BehaviorCloningBaseline, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# 4. Configurar entrenamiento
hidden_dim = 128
model_bc = BehaviorCloningBaseline(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model_bc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

EPOCHS = 50
print(f"\n🚀 Iniciando entrenamiento por {EPOCHS} épocas...")

for epoch in range(EPOCHS):
    model_bc.train()
    total_loss = 0
    correct_preds = 0
    total_samples = 0
    
    for batch_states, batch_actions, batch_masks in dataloader:
        # Aplanar la secuencia (el MLP evalúa cada instante de forma independiente)
        states_flat = batch_states.view(-1, input_dim)
        actions_flat = batch_actions.view(-1).long()
        mask_flat = batch_masks.view(-1)
        
        # Filtrar el padding (entrenar solo con datos reales, donde mask == 1)
        valid_idx = mask_flat > 0
        x = states_flat[valid_idx]
        y_true = actions_flat[valid_idx]
        
        if len(x) == 0: continue
            
        # Forward pass
        optimizer.zero_grad()
        logits = model_bc(x)
        loss = criterion(logits, y_true)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Métricas
        total_loss += loss.item()
        _, y_pred = torch.max(logits, 1)
        correct_preds += (y_pred == y_true).sum().item()
        total_samples += y_true.size(0)
        
    avg_loss = total_loss / len(dataloader)
    accuracy = (correct_preds / total_samples) * 100
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

# 5. GUARDAR EL MODELO ENTRENADO
model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'modelo_baseline_bc.pth')

torch.save(model_bc.state_dict(), model_save_path)
print(f"\n💾 ¡Modelo entrenado y guardado exitosamente en:\n{model_save_path}")