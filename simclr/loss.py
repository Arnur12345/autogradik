"""
NT-Xent: the normalized temperature-scaled cross-entropy loss (SimCLR Eq. 1).
                                                        ***TASK: fill in the TODO.***

A batch of N images gives two views each -> 2N projections. For each anchor, its positive
is the other view of the same image; the other 2N-2 projections are negatives. You L2-
normalize the projections (so dot product = cosine similarity), scale by 1/temperature,
mask out each row's self-similarity, and apply cross-entropy toward the positive index.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """z1, z2: [N, D] projections of the two views. Returns a scalar loss.

        TODO:
        """
        z = nn.functional.normalize(torch.concat([z1, z2], dim=0))
        b, emb = z.shape
        S = (z * z.T) / self.temperature
        
        S.fill_diagonal_

        loss = 1/2 * (F.cross_entropy(S[:b//2, :], torch.arange(b//2, b)) + 
                      F.cross_entropy(S[b:, :], torch.arange(0, b//2)))

        return loss
