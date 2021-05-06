from torch import nn


class MLP(nn.Module):

    def __init__(self, input_dim=128, hidden_size=256, output_dim=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SiameseArm(nn.Module):

    def __init__(
            self,
            encoder: nn.Module = None,
            input_dim: int = 16,
            hidden_size: int = 32,
            output_dim: int = 16,
    ):
        super().__init__()
        # Encoder
        self.encoder = encoder
        # Projector
        self.projector = MLP(input_dim=input_dim, hidden_size=hidden_size, output_dim=hidden_size)
        # Predictor
        self.predictor = MLP(input_dim=hidden_size, hidden_size=hidden_size, output_dim=output_dim)

    def forward(self, x):
        y = self.encoder(x)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h
