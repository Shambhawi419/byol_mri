"""
byolloss.py
-----------
BYOL Loss: Bootstrap Your Own Latent
Paper: https://arxiv.org/abs/2006.07733

Key difference from SimCLR (SupConLoss):
- No negative pairs needed
- Simply minimizes MSE between online prediction and target projection
- Works well with small datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BYOLLoss(nn.Module):
    """
    BYOL Loss function.

    Takes the online network's prediction and the target network's
    projection and minimizes the MSE between them.

    Loss = 2 - 2 * cosine_similarity(online_pred, target_proj)
         = MSE between L2-normalized vectors
    """

    def __init__(self):
        super(BYOLLoss, self).__init__()

    def forward(self, online_pred: torch.Tensor,
                target_proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            online_pred: prediction from online network [bsz, proj_dim]
            target_proj: projection from target network [bsz, proj_dim]
        Returns:
            scalar loss
        """
        online_pred = F.normalize(online_pred, dim=-1, p=2, eps=1e-8)
        target_proj = F.normalize(target_proj, dim=-1, p=2, eps=1e-8)
        online_pred = torch.nan_to_num(online_pred, nan=0.0)
        target_proj = torch.nan_to_num(target_proj, nan=0.0)
        loss = 2 - 2 * (online_pred * target_proj).sum(dim=-1)
        return loss.mean()