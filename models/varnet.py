# ─────────────────────────────────────────────────────────────────
# BYOL Addition — ProjectionMLP, PredictionMLP, VarNetBYOL
# ─────────────────────────────────────────────────────────────────

class ProjectionMLP(nn.Module):
    """
    Projector MLP used in BYOL.
    Maps VarNet encoder output to a compact latent vector.
    Both online and target networks share this architecture.

    Input:  VarNet output (b, coils, h, w, 2)
    Output: projection vector (b, proj_dim)
    """
    def __init__(self, in_dim: int = 2,
                 hidden_dim: int = 4096,
                 out_dim: int = 256):
        super(ProjectionMLP, self).__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, coils, h, w, 2) -> collapse coils -> (b, h, w, 2)
        x = x.mean(dim=1)
        # (b, h, w, 2) -> (b, 2, h, w) for AdaptiveAvgPool2d
        x = x.permute(0, 3, 1, 2)
        return self.net(x)


class PredictionMLP(nn.Module):
    """
    Predictor MLP used in BYOL.
    Only the ONLINE network has this — this asymmetry between
    online and target networks prevents representational collapse
    without needing negative pairs.

    Input:  projection vector (b, proj_dim)
    Output: prediction vector (b, proj_dim)
    """
    def __init__(self, in_dim: int = 256,
                 hidden_dim: int = 4096,
                 out_dim: int = 256):
        super(PredictionMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VarNetBYOL(nn.Module):
    """
    VarNet encoder wrapped with BYOL projection and prediction heads.

    Architecture:
        Online  network: VarNet encoder → ProjectionMLP → PredictionMLP
        Target network: VarNet encoder → ProjectionMLP (no predictor)

    The asymmetry (predictor only on online) is the key mechanism
    that prevents representational collapse in BYOL.

    Args:
        varnet:        pretrained or randomly initialized VarNet backbone
        use_predictor: True for online network, False for target network
        proj_dim:      output dimension of projection/prediction MLPs
        hidden_dim:    hidden dimension of projection/prediction MLPs
    """
    def __init__(self, varnet: nn.Module,
                 use_predictor: bool = True,
                 proj_dim: int = 256,
                 hidden_dim: int = 4096):
        super(VarNetBYOL, self).__init__()
        self.encoder = varnet
        self.projector = ProjectionMLP(in_dim=2,
                                       hidden_dim=hidden_dim,
                                       out_dim=proj_dim)
        self.use_predictor = use_predictor
        if use_predictor:
            self.predictor = PredictionMLP(in_dim=proj_dim,
                                           hidden_dim=hidden_dim,
                                           out_dim=proj_dim)

    def forward(self,
                masked_kspace: torch.Tensor,
                mask: torch.Tensor,
                num_low_frequencies: int) -> torch.Tensor:
        """
        Args:
            masked_kspace:      undersampled k-space (b, coils, h, w, 2)
            mask:               sampling mask (b, 1, 1, w, 1)
            num_low_frequencies: number of low frequency lines kept
        Returns:
            prediction (online) or projection (target) vector (b, proj_dim)
        """
        # encode through VarNet
        enc_out = self.encoder(masked_kspace=masked_kspace,
                               mask=mask,
                               num_low_frequencies=num_low_frequencies)
        # project to latent space
        proj = self.projector(enc_out)

        # predict (online network only)
        if self.use_predictor:
            return self.predictor(proj)
        return proj