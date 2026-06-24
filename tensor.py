import numpy as np 
from typing import TypeAlias

Shape: TypeAlias = tuple[int, ...]

class Tensor:
    def __init__(self, data: list | np.ndarray, _children=(), _op='', dtype=np.float32) -> None:
        self.data: np.array = np.array(data, dtype=dtype)
        self.grad: np.ndarray = np.zeros_like(self.data, dtype=dtype)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self_grad = out.grad
            other_grad = out.grad
            
            if self_grad.shape != self.data.shape:
                self_grad = self_grad.sum(axis=tuple(range(len(self_grad.shape) - len(self.data.shape))))
            if other_grad.shape != other.data.shape:
                other_grad = other_grad.sum(axis=tuple(range(len(other_grad.shape) - len(other.data.shape))))
                
            self.grad += self_grad
            other.grad += other_grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += out.grad * (out.data > 0)
        
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'only pow and ints'
        try:
            out = Tensor(self.data ** other, (self, other), 'f**{other}')
        except:
            print("Data types for power are not Int or Float")

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        
        out._backward = _backward
        return out

    def __sum__(self):
        out = Tensor(np.sum(self.data), (self,), 'sum')

        def _backward():
            self.grad += np.broadcast_to(out.grad, self.data.shape)
        
        out._backward = _backward
        return out

    def __mean__(self):
        out = Tensor(np.mean(self.data), (self,), 'mean')

        def _backward():
            n = self.data.size
            self.grad += np.broadcast_to(out.grad, self.data.shape) / n

        out._backward = _backward
        return out

    def backward(self):
        topology = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topology.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topology):
            v._backward()

        