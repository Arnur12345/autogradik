import numpy as np
from tensor import Tensor

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        bound = np.sqrt(2.0 / in_features)
        self.W = Tensor(np.random.randn(in_features, out_features) * bound)
        self.b = Tensor(np.zeros(out_features))

    def __call__(self, X: Tensor) -> Tensor:
        return (X @ self.W) + self.b

    def parameters(self):
        return [self.W, self.b]

class CrossEntropyLoss(Module):
    def __call__(self, logits, target):
        N = logits.data.shape[0]

        #softmax
        z = logits.data
        z_shift = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shift)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        #loss
        log_probs = -np.log(probs[range(N), target] + 1e-7)
        loss = np.mean(log_probs)

        out = Tensor(loss, (logits,), 'CrossEntropyLoss')

        def _backward():
            dlogits = probs.copy()
            dlogits[range(N), target] -= 1
            dlogits = dlogits / N 
            logits.grad += out.grad * dlogits

        out._backward = _backward
        return out

