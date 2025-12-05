import torch
from torch import nn
from .augmentations import GaussianSmoothing


# -------------------------------------------------------
# Minimal TCN Residual Block (light, safe for GPU)
# -------------------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1, dropout=0.0):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size,
            padding=padding, dilation=dilation
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out + self.residual(x)


# -------------------------------------------------------
# GRU + TCN Hybrid Decoder
# -------------------------------------------------------
class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=32,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(GRUDecoder, self).__init__()

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional

        # ---------- preprocessing layers ----------
        self.inputLayerNonlinearity = nn.Softsign()
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )

        # Day transforms
        self.dayWeights = nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x] = torch.eye(neural_dim)

        # Per-day input layers
        for x in range(nDays):
            setattr(self, f"inpLayer{x}", nn.Linear(neural_dim, neural_dim))
            layer = getattr(self, f"inpLayer{x}")
            layer.weight = nn.Parameter(layer.weight + torch.eye(neural_dim))

        # ---------- NEW: TCN FRONT-END ----------
        self.tcn = nn.Sequential(
            TemporalBlock(neural_dim, neural_dim, kernel_size=5, dilation=1, dropout=0.2),
            TemporalBlock(neural_dim, neural_dim, kernel_size=5, dilation=2, dropout=0.2),
            TemporalBlock(neural_dim, neural_dim, kernel_size=5, dilation=4, dropout=0.2),
            TemporalBlock(neural_dim, neural_dim, kernel_size=5, dilation=8, dropout=0.2), # addition for 2.3, also reduce all dropouts
        )

        # ---------- kernel/stride unfolding ----------
        self.unfolder = nn.Unfold(
            (self.kernelLen, 1), stride=self.strideLen
        )

        # ---------- GRU ----------
        self.gru_decoder = nn.GRU(
            neural_dim * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # ---------- output head ----------
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc_decoder_out = nn.Linear(out_dim, n_classes + 1)

    # -------------------------------------------------------
    # Forward
    # -------------------------------------------------------
    def forward(self, neuralInput, dayIdx):
        # (B, T, C)
        neuralInput = neuralInput.transpose(1, 2)   # → (B, C, T)
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = neuralInput.transpose(1, 2)   # → (B, T, C)

        # Day transform
        W = torch.index_select(self.dayWeights, 0, dayIdx)
        B = torch.index_select(self.dayBias, 0, dayIdx)
        x = torch.einsum("btd,bdk->btk", neuralInput, W) + B
        x = self.inputLayerNonlinearity(x)

        # ----- NEW: TCN -----
        x = x.transpose(1, 2)   # → (B, C, T)
        x = self.tcn(x)
        x = x.transpose(1, 2)   # → (B, T, C)

        # Stride/kernel unfolding
        x = self.unfolder(
            x.transpose(1, 2).unsqueeze(3)
        ).transpose(1, 2)

        # GRU
        num_layers = self.layer_dim * (2 if self.bidirectional else 1)
        h0 = torch.zeros(num_layers, x.size(0), self.hidden_dim, device=self.device)
        hid, _ = self.gru_decoder(x, h0)

        # Linear head
        return self.fc_decoder_out(hid)
