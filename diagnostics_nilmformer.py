import numpy as np
import torch
import sys
sys.path.insert(0, './transformer')
from nilmformer_paper import create_nilmformer_paper

# Paths
p = 'c:/Users/gamek/School/TeamProject/MTS3-MCTE-Team-Project-Energy-G1/data/processed/1sec_new/model_ready'

# 1. Data shapes and overlap
Xtr = np.load(f'{p}/X_train.npy', mmap_mode='r')
Xval = np.load(f'{p}/X_val.npy', mmap_mode='r')
ytr = np.load(f'{p}/y_train.npy', mmap_mode='r')
yval = np.load(f'{p}/y_val.npy', mmap_mode='r')

print('X_train:', Xtr.shape, 'X_val:', Xval.shape)
print('y_train:', ytr.shape, 'y_val:', yval.shape)

# 2. Data leakage check (full val in train)
leak = False
for i in range(min(500, Xval.shape[0])):
    for j in range(min(500, Xtr.shape[0])):
        if np.allclose(Xval[i], Xtr[j]):
            print(f'LEAK: val[{i}] == train[{j}]')
            leak = True
print('Any data leakage:', leak)

# 3. Target stats
print('y_train min/max/mean:', ytr.min(), ytr.max(), ytr.mean())
print('y_val min/max/mean:', yval.min(), yval.max(), yval.mean())

# 4. Model output sanity check
apps = ['HeatPump','Dishwasher','WashingMachine','Dryer','Oven','Stove','RangeHood','EVCharger','EVSocket','GarageCabinet','RainwaterPump']
model = create_nilmformer_paper(appliances=apps, c_embedding=6, d_model=48, n_layers=2, n_head=4, multi_appliance=True)
model.eval()
with torch.no_grad():
    out = model(torch.FloatTensor(Xtr[:8]))
    for k,v in out.items():
        print(f'{k}: min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.6f}')

# 5. BatchNorm/InstanceNorm mode
for m in model.modules():
    if isinstance(m, torch.nn.BatchNorm1d):
        print('BatchNorm1d running stats:', m.running_mean[:5].cpu().numpy())

# 6. Gradient flow check (single batch)
model.train()
x = torch.FloatTensor(Xtr[:8])
y = torch.FloatTensor(ytr[:8])
y_dict = {apps[i]: y[:,:,i:i+1] for i in range(len(apps))}
output = model(x)
loss = sum([(output[a] - y_dict[a]).abs().mean() for a in apps])
loss.backward()
grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
print('Gradient norms:', grad_norms)
print('Any zero grad:', any([g == 0 for g in grad_norms]))
