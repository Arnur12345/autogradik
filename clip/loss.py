"""
CLIP's symmetric contrastive loss (Radford et al., 2021, Figure 3), adapted to labels.
                                                        ***TASK: fill in the TODO.***

In vanilla CLIP each image pairs with exactly one text, so the target is the diagonal
(image i matches text i). Here the "text" is a class label, and a batch can contain several
images of the same class -- which all share one class embedding. So the positive of image i
is every column j with the same label.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import VisionTransformer, ClassEncoder


class CLIPLoss(nn.Module):
    def forward(self, image_emb, class_emb, labels, logit_scale):
        """image_emb, class_emb: [N, D]; labels: [N]; logit_scale: scalar param.

        TODO:

        This same target works for both batch regimes: with unique-class batches
        (dataset one_per_class=True) `same` is the identity, so it becomes the diagonal.
        """
        image_emb = F.normalize(image_emb, p=2, dim=1)
        class_emb = F.normalize(class_emb, p=2, dim=1)
        
        logits = torch.matmul(image_emb, class_emb.t()) * torch.exp(logit_scale)

        arange = torch.arange(logits.shape[0], device=logits.device)
        
        loss_i = F.cross_entropy(logits, arange)
        loss_t = F.cross_entropy(logits.t(), arange)
        
        loss = (loss_i + loss_t) / 2

        return loss