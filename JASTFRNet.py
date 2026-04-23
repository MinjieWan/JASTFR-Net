import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class JointAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        mid = max(channels // reduction, 4)

        # Channel branch
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1)
        )

        # Spatial branch
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1)
        )

        self.fusion = nn.Conv2d(2 * channels, channels, 1)

        self.fusion = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):

        B, C, H, W = x.shape

        ch = self.channel_att(x)      # B,C,1,1
        sp = self.spatial_att(x)      # B,1,H,W

        ch = ch.expand(-1, -1, H, W)  # B,C,H,W
        sp = sp.expand(-1, C, -1, -1)  # B,C,H,W

        fusion = torch.cat([ch, sp], dim=1)  # B,2C,H,W

        att = torch.sigmoid(self.fusion(fusion))

        return x * att


class AttentionConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=(3, 3), bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = (self.kernel_size[0]//2, self.kernel_size[1]//2)
        self.bias = bias
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4*hidden_dim, kernel_size, padding=self.padding, bias=bias)
        self.att = JointAttention(4 * hidden_dim)

    def forward(self, x, state):
        if state == (None, None):
            B, _, H, W = x.size()
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        gates = self.att(gates)

        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class JAConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.lstm_cell = AttentionConvLSTMCell(in_channels, hidden_channels, kernel_size)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        h, c = None, None
        outputs = []
        for i in range(T):
            h, c = self.lstm_cell(x[:, i], (h, c))
            outputs.append(h)
        return torch.stack(outputs, dim=1)  # [B, T, hidden, H, W]


class Encoder3D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv3d(32, 48, kernel_size=3, stride=(1,2,2), padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv3d(48, 64, kernel_size=3, stride=(1,2,2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return [f1, f2, f3]


class TemporalWeightedAggregation(nn.Module):
    def __init__(self, T):
        super().__init__()
        # Initialized to 1, equivalent to doing mean at the beginning
        self.weight = nn.Parameter(torch.ones(T))

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        return: [B, C, H, W]
        """
        w = torch.softmax(self.weight, dim=0)  # [T]
        return (x * w.view(1, 1, -1, 1, 1)).sum(dim=2)


class Decoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = self._block(64+48, 48)
        self.up2 = self._block(48+32, 32)
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)  # Randomly drop some spatial features to enhance generalization
        )

    def forward(self, feats, orig_size):
        f1, f2, f3 = feats
        x = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.up1(torch.cat([x, f2], dim=1))
        x = self.up2(torch.cat([x, f1], dim=1))
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        return self.out_conv(x)


class EdgeRefine(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        self.refine = nn.Conv2d(2, 1, 3, padding=1)

    def forward(self, pred, raw):
        edge_h = torch.abs(raw[:, :, 1:, :] - raw[:, :, :-1, :])
        edge_h = F.pad(edge_h, (0, 0, 0, 1))
        edge_w = torch.abs(raw[:, :, :, 1:] - raw[:, :, :, :-1])
        edge_w = F.pad(edge_w, (0, 1, 0, 0))
        edge = (edge_h + edge_w) / 2
        edge = edge.mean(dim=1, keepdim=True)
        if edge.shape[-2:] != pred.shape[-2:]:
            edge = F.interpolate(edge, size=pred.shape[-2:], mode='bilinear', align_corners=False)
        fusion = torch.cat([pred, edge], dim=1)
        weight = self.att(fusion)
        return self.refine(fusion * weight)


# overall network
class JASTFRNet(nn.Module):
    def __init__(self, in_channels=1, hidden=8, timesteps=5):
        super().__init__()
        self.timesteps = timesteps
        self.lstm = JAConvLSTM(in_channels, hidden)
        self.encoder = Encoder3D(in_ch=hidden)
        self.temp_agg = TemporalWeightedAggregation(timesteps)
        self.decoder = Decoder2D()
        self.edge_refine = EdgeRefine()

    def forward(self, x):
        orig = x  # [B,C,T,H,W]
        x = x.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W]
        lstm_out = self.lstm(x)  # [B,T,C,H,W]
        feat3d = lstm_out.permute(0, 2, 1, 3, 4)  # [B,hidden,T,H,W]

        feats = self.encoder(feat3d)

        f1, f2, f3 = feats
        f1 = self.temp_agg(f1)
        f2 = self.temp_agg(f2)
        f3 = self.temp_agg(f3)

        out = self.decoder([f1, f2, f3], (x.size(-2), x.size(-1)))

        raw_last = orig[:, :, -1, :, :]
        refined = self.edge_refine(out, raw_last)
        return refined


class HybridLoss(nn.Module):
    def __init__(self, pos_weight=5.0, alpha=0.7, beta=0.3, gamma=0.75):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def forward(self, pred, target):
        # --- Ensure the input is [B, 1, H, W] ---
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)

        pred_prob = torch.sigmoid(pred)
        smooth = 1e-5

        # BCE
        bce = self.bce(pred, target)

        # Dice
        intersection = (pred_prob * target).sum()
        dice = 1 - (2. * intersection + smooth) / (pred_prob.sum() + target.sum() + smooth)

        # Tversky
        tp = (pred_prob * target).sum()
        fp = ((1 - target) * pred_prob).sum()
        fn = (target * (1 - pred_prob)).sum()
        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        focal_tversky = (1 - tversky) ** self.gamma

        ssim_loss = 1 - ssim(pred_prob, target, data_range=1.0, size_average=True)

        return 0.5 * bce + 0.3 * dice + 0.15 * focal_tversky + 0.05 * ssim_loss

