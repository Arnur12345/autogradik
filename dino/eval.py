"""
kNN evaluation of the learned representation (no labels used during training).
 
We build a memory bank of L2-normalized features h = f(x) over the train set, then for
each test image retrieve its k nearest neighbours by cosine similarity and predict the
label by a temperature-weighted vote. This is the standard weighted-kNN monitor
(Wu et al., 2018) used to track SSL representation quality cheaply.
"""
import torch
import torch.nn.functional as F
 
 
@torch.no_grad()
def _extract_features(model, loader, device):
    """L2-normalized features and labels for every image in `loader`."""
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        h = model.features(x.to(device, non_blocking=True))
        feats.append(F.normalize(h, dim=1))
        labels.append(y.to(device, non_blocking=True))
    return torch.cat(feats), torch.cat(labels)
 
 
@torch.no_grad()
def knn_evaluate(model, memory_loader, test_loader, device,
                 k=200, temperature=0.1, num_classes=100):
    """Top-1 accuracy (%) of a weighted kNN classifier over the frozen features."""
    bank_feats, bank_labels = _extract_features(model, memory_loader, device)   # [M, D], [M]
 
    correct = total = 0
    for x, y in test_loader:
        y = y.to(device, non_blocking=True)
        q = F.normalize(model.features(x.to(device, non_blocking=True)), dim=1)  # [B, D]
 
        sim = q @ bank_feats.t()                                # [B, M] cosine similarity
        sim_w, idx = sim.topk(k, dim=1)                         # top-k neighbours
        neighbour_labels = bank_labels[idx]                     # [B, k]
        weights = (sim_w / temperature).exp()                   # [B, k] closer -> larger vote
 
        votes = torch.zeros(y.size(0), num_classes, device=device)
        votes.scatter_add_(1, neighbour_labels, weights)        # [B, C] weighted class scores
        pred = votes.argmax(dim=1)
 
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total