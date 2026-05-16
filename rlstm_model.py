"""
HMR-BiLSTM: Bidirectional LSTM with Hybrid Memory Control
========================================================

Kiến trúc: Hybrid Memory Path
- RMC path: Softmax attention blend của c_keep và c_add
- LSTM path: Standard memory update f_t * c_prev + i_t * g_t
- Blend qua learnable beta gate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#  CNN FEATURE EXTRACTOR
# =============================================================================
class ECGFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, output_channels=64, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, output_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(output_channels)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        return x.transpose(1, 2)


# =============================================================================
#  ATTENTION POOLING
# =============================================================================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, h_seq):
        scores = self.attention(h_seq)
        weights = F.softmax(scores, dim=1)
        pooled = (h_seq * weights).sum(dim=1)
        return pooled, weights.squeeze(-1)


# =============================================================================
#  HMR-BiLSTM CELL — Phương án B (Hybrid Memory Path)
# =============================================================================
class RLSTMCell(nn.Module):
    """
    HMR-BiLSTM cell với Hybrid Memory Path:
    - RMC path: softmax attention blend của c_keep và c_add
    - LSTM path: standard memory update f_t * c_prev + i_t * g_t
    - Blend qua learnable beta gate

    Eq. (12) phiên bản B (giống A):
        scores = W_alpha [c_keep; c_add]
        alpha  = softmax(scores)
    Eq. (13a) RMC path:
        c_rmc = alpha^(1) * c_keep + alpha^(2) * c_add
    Eq. (13b) LSTM path (MỚI):
        c_lstm = f_t * c_prev + i_t * g_t
    Eq. (13c) Blend gate (MỚI):
        beta = sigmoid(W_beta h_prev)
    Eq. (13d) Hybrid output (MỚI):
        c_t = beta * c_rmc + (1 - beta) * c_lstm
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # --- Bốn cổng LSTM chuẩn — Eq. (2)-(5) ---
        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size)

        # --- Residual Memory Control — Eq. (8)-(9) ---
        self.W_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_h_rmc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # --- Giải pháp 1: Learnable softmax scoring — Eq. (12) ---
        self.W_alpha = nn.Linear(2 * hidden_size, 2)

        # --- Giải pháp 2: Blend gate β — Eq. (13c) MỚI ---
        # β điều khiển blend giữa RMC path và LSTM path
        # Element-wise gate (size H) cho phép từng dimension blend khác nhau
        self.W_beta = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        # Lưu trữ cho interpretability
        self.last_r_t = None
        self.last_c_keep = None
        self.last_c_add = None
        self.last_alpha = None
        self.last_beta = None   # MỚI: blend gate
        self.last_c_rmc = None  # MỚI: RMC path output
        self.last_c_lstm = None # MỚI: LSTM path output

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                if any(k in name for k in ["W_h", "W_c", "W_h_rmc"]):
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                if "W_x" in name or "W_h" in name:
                    n = param.size(0) // 4
                    with torch.no_grad():
                        param[n:2*n].fill_(1.0)
                # CRITICAL: bias của W_beta khởi tạo âm để bắt đầu β ≈ 0
                # → mô hình bắt đầu như LSTM chuẩn, sau đó học dùng RMC
                if "W_beta.bias" in name:
                    with torch.no_grad():
                        param.fill_(-1.0)   # sigmoid(-1) ≈ 0.27

    def forward(self, x_t, h_prev, c_prev):
        # === Bước 1: Bốn cổng LSTM chuẩn — Eq. (2)-(5) ===
        gates = self.W_x(x_t) + self.W_h(h_prev)
        i_t, f_t, o_t, g_t = torch.split(gates, self.hidden_size, dim=1)
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        o_t = torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)

        # === Bước 2: Interaction Term — Eq. (8) ===
        Wc_c = self.W_c(c_prev)
        Wh_h = self.W_h_rmc(h_prev)
        m_t = Wc_c * Wh_h

        # === Bước 3: Residual Gate — Eq. (9) ===
        linear_part = self.layer_norm(Wc_c + Wh_h)
        r_t = torch.sigmoid(linear_part + 0.1 * m_t)

        # === Bước 4: Memory Decomposition — Eq. (10), (11) ===
        c_keep = (f_t + r_t * (1.0 - f_t)) * c_prev
        c_add  = (1.0 - r_t) * (i_t * g_t)

        # === Bước 5a: RMC PATH — softmax attention (Giải pháp 1) ===
        combined = torch.cat([c_keep, c_add], dim=-1)
        scores = self.W_alpha(combined)
        alpha = F.softmax(scores, dim=-1)
        alpha_keep = alpha[:, 0:1]
        alpha_add  = alpha[:, 1:2]
        c_rmc = alpha_keep * c_keep + alpha_add * c_add

        # === Bước 5b: LSTM PATH (MỚI - Giải pháp 2) ===
        # Standard LSTM behavior - đảm bảo "safety net"
        c_lstm = f_t * c_prev + i_t * g_t

        # === Bước 5c: Blend β qua learnable gate (MỚI - Giải pháp 2) ===
        beta = torch.sigmoid(self.W_beta(h_prev))   # (B, H) ∈ (0, 1)

        # === Bước 5d: Hybrid output ===
        c_t = beta * c_rmc + (1.0 - beta) * c_lstm

        # === Bước 6: Hidden state — Eq. (14) ===
        h_t = o_t * torch.tanh(c_t)

        # Lưu nội tại
        self.last_r_t = r_t
        self.last_c_keep = c_keep
        self.last_c_add = c_add
        self.last_alpha = alpha
        self.last_beta = beta       # MỚI
        self.last_c_rmc = c_rmc     # MỚI
        self.last_c_lstm = c_lstm   # MỚI

        return h_t, c_t


# =============================================================================
#  HMR-BiLSTM LAYER
# =============================================================================
class RLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = RLSTMCell(input_size, hidden_size, dropout=dropout)

    def forward(self, x, h0=None, c0=None):
        B, T, _ = x.size()
        device = x.device

        h = torch.zeros(B, self.hidden_size, device=device) if h0 is None else h0
        c = torch.zeros(B, self.hidden_size, device=device) if c0 is None else c0

        h_outputs, r_outputs, ck_outputs, ca_outputs = [], [], [], []

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            h_outputs.append(h)
            r_outputs.append(self.cell.last_r_t.clone())
            ck_outputs.append(self.cell.last_c_keep.clone())
            ca_outputs.append(self.cell.last_c_add.clone())

        h_seq      = torch.stack(h_outputs,  dim=1)
        r_seq      = torch.stack(r_outputs,  dim=1)
        c_keep_seq = torch.stack(ck_outputs, dim=1)
        c_add_seq  = torch.stack(ca_outputs, dim=1)

        return h_seq, (h, c), r_seq, c_keep_seq, c_add_seq


# =============================================================================
#  BIDIRECTIONAL HMR-BiLSTM
# =============================================================================
class BiRLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.forward_layer  = RLSTMLayer(input_size, hidden_size, dropout=dropout)
        self.backward_layer = RLSTMLayer(input_size, hidden_size, dropout=dropout)

    def forward(self, x):
        h_fwd, (h_T_fwd, c_T_fwd), r_fwd, ck_fwd, ca_fwd = self.forward_layer(x)

        x_reversed = torch.flip(x, dims=[1])
        h_bwd, (h_T_bwd, c_T_bwd), r_bwd, ck_bwd, ca_bwd = \
            self.backward_layer(x_reversed)

        h_bwd  = torch.flip(h_bwd,  dims=[1])
        r_bwd  = torch.flip(r_bwd,  dims=[1])
        ck_bwd = torch.flip(ck_bwd, dims=[1])
        ca_bwd = torch.flip(ca_bwd, dims=[1])

        h_seq = torch.cat([h_fwd, h_bwd], dim=-1)
        h_T   = torch.cat([h_T_fwd, h_T_bwd], dim=-1)
        c_T   = torch.cat([c_T_fwd, c_T_bwd], dim=-1)

        return h_seq, (h_T, c_T), r_fwd, r_bwd, ck_fwd, ck_bwd, ca_fwd, ca_bwd


# =============================================================================
#  CLASSIFIER với CNN + HMR-BiLSTM + ATTENTION POOLING
# =============================================================================
class RLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 96,
        dropout: float = 0.25,
        num_classes: int = 5,
        cnn_out_channels: int = 64,
    ):
        super().__init__()
        self.cnn = ECGFeatureExtractor(
            input_channels=input_size,
            output_channels=cnn_out_channels,
            dropout=dropout * 0.5,
        )
        self.birlstm = BiRLSTM(cnn_out_channels, hidden_size, dropout=dropout)
        self.attention_pool = AttentionPooling(2 * hidden_size)
        self.layer_norm = nn.LayerNorm(2 * hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, return_internals=False):
        features = self.cnn(x)
        outputs = self.birlstm(features)
        h_seq = outputs[0]
        r_fwd, r_bwd = outputs[2], outputs[3]

        h_pooled, attn_weights = self.attention_pool(h_seq)
        h_pooled = self.layer_norm(h_pooled)
        h_pooled = self.dropout(h_pooled)
        logits = self.classifier(h_pooled)

        if return_internals:
            internals = {
                "r_fwd": r_fwd,
                "r_bwd": r_bwd,
                "c_keep_fwd": outputs[4],
                "c_keep_bwd": outputs[5],
                "c_add_fwd":  outputs[6],
                "c_add_bwd":  outputs[7],
                "attention_weights": attn_weights,
            }
            return logits, internals
        return logits


# =============================================================================
#  LOSS FUNCTIONS
# =============================================================================
def temporal_smoothness_loss(r_seq):
    if r_seq.size(1) < 2:
        return torch.tensor(0.0, device=r_seq.device, requires_grad=True)
    diff = r_seq[:, 1:, :] - r_seq[:, :-1, :]
    squared_norm = (diff ** 2).sum(dim=-1)
    return squared_norm.mean()


class RLSTMLoss(nn.Module):
    def __init__(self, lambda_smooth=0.01, class_weights=None):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.task_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, targets, r_fwd=None, r_bwd=None):
        l_task = self.task_loss(logits, targets)
        l_smooth = torch.tensor(0.0, device=logits.device)
        if r_fwd is not None:
            l_smooth = l_smooth + temporal_smoothness_loss(r_fwd)
        if r_bwd is not None:
            l_smooth = l_smooth + temporal_smoothness_loss(r_bwd)
        if r_fwd is not None and r_bwd is not None:
            l_smooth = l_smooth / 2.0
        total = l_task + self.lambda_smooth * l_smooth
        return total, {
            "task": l_task.detach(),
            "smooth": l_smooth.detach(),
            "total": total.detach(),
        }


# =============================================================================
#  DEMO
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, D = 8, 187, 1
    model = RLSTMClassifier(input_size=D, hidden_size=96, num_classes=5)
    x = torch.randn(B, T, D)
    y = torch.randint(0, 5, (B,))
    criterion = RLSTMLoss(lambda_smooth=0.003)

    logits, internals = model(x, return_internals=True)
    print(f"[Plan B] Logits shape: {logits.shape}")
    print(f"[Plan B] r_fwd shape:  {internals['r_fwd'].shape}")
    print(f"[Plan B] Tổng tham số: {sum(p.numel() for p in model.parameters()):,}")

    loss, comp = criterion(logits, y, internals["r_fwd"], internals["r_bwd"])
    print(f"[Plan B] Loss: {loss.item():.4f}")
    loss.backward()
    print("[Plan B] ✓ Backward OK")
