import numpy as np
from layers import Module, CrossEntropyLoss, Linear
from tensor import Tensor
import torch
from torch import nn 

torch.manual_seed(42)

X_np = np.random.randn(4, 8).astype(np.float32)
y_np = np.random.randint(0, 3, size=(4,))

print("--- Models Initialization ---")
my_l1 = Linear(8, 16)
my_l2 = Linear(16, 3)
my_criterion = CrossEntropyLoss()

torch_model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)
torch_criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    torch_model[0].weight.copy_(torch.tensor(my_l1.W.data.T))
    torch_model[0].bias.copy_(torch.tensor(my_l1.b.data))
    torch_model[2].weight.copy_(torch.tensor(my_l2.W.data.T))
    torch_model[2].bias.copy_(torch.tensor(my_l2.b.data))

X_torch = torch.tensor(X_np, requires_grad=True)
y_torch = torch.tensor(y_np, dtype=torch.long)

my_X = Tensor(X_np)
my_h1 = my_l1(my_X).relu()
my_logits = my_l2(my_h1)
my_loss = my_criterion(my_logits, y_np)

torch_logits = torch_model(X_torch)
torch_loss = torch_criterion(torch_logits, y_torch)

print(f"Your Loss:  {my_loss.data:.8f}")
print(f"Torch Loss: {torch_loss.item():.8f}")

loss_diff = abs(my_loss.data - torch_loss.item())
print(f"Loss Difference: {loss_diff:.8f}")
assert loss_diff < 1e-5, "Loss mismatch on forward pass!"

my_l1.zero_grad()
my_l2.zero_grad()
torch_model.zero_grad()

my_loss.backward()
torch_loss.backward()

print("\n--- Gradient Verification (Backward) ---")
w2_grad_diff = np.max(np.abs(my_l2.W.grad - torch_model[2].weight.grad.numpy().T))
print(f"Max dW2 diff: {w2_grad_diff:.8e}")

b2_grad_diff = np.max(np.abs(my_l2.b.grad - torch_model[2].bias.grad.numpy()))
print(f"Max db2 diff: {b2_grad_diff:.8e}")

w1_grad_diff = np.max(np.abs(my_l1.W.grad - torch_model[0].weight.grad.numpy().T))
print(f"Max dW1 diff: {w1_grad_diff:.8e}")

b1_grad_diff = np.max(np.abs(my_l1.b.grad - torch_model[0].bias.grad.numpy()))
print(f"Max db1 diff: {b1_grad_diff:.8e}")

all_diffs = [w2_grad_diff, b2_grad_diff, w1_grad_diff, b1_grad_diff]
assert max(all_diffs) < 1e-5, "Gradients mismatch!"
print("\nVALIDATION SUCCESSFUL! Your framework matches PyTorch exactly.")