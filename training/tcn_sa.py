"""
WaveNILM v6 - TCN + Flash Attention Architecture (SOTA Real-Time NILM)
======================================================================

Architettura causale per Non-Intrusive Load Monitoring in tempo reale.
Ogni predizione usa SOLO dati passati → deployment streaming ready.

Riferimenti:
- "Spatio-temporal attention fusion network for NILM" (Zhang et al., 2025)
- "TCAN: Temporal Convolutional Attention Network" (2021-2024)
- "Causal TCN-Attention Network for Real-time Load Disaggregation" (Liao et al., 2024)

Miglioramenti rispetto a v5 (TCN puro):
1. Flash Attention: F.scaled_dot_product_attention(is_causal=True) - O(T) memory, 10x faster
2. Multi-Scale Inception: kernel 1,3,5,7 paralleli → preserva magnitudine segnale
3. Deep Power Head: 3-layer MLP + GELU + BatchNorm → regressione potenza accurata
4. Skip Connections: stem → mid → final fusion per multi-resolution features
5. Gated Output: power = gate × softplus(raw) per coerenza ON/OFF

Configurazione SOTA:
- Window: 4096 samples (68 min @ 1Hz)
- Stride: 512 (training) / 1 (inference real-time)
- n_blocks: 12 → Receptive Field = 4101 ≥ 4096
- Hidden: 64 channels → 192K parameters

Garanzie Causalità:
- CausalConv1d: left-padding = (kernel-1) × dilation
- Flash Attention: is_causal=True → maschera triangolare automatica
- Output: h = x[:, -1, :] → predice solo ultimo timestep

Performance attese: F1 > 0.90, MAE_ON reduction 18-22% vs TCN vanilla
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpatialDropout1D(nn.Module):
    """Channel-wise dropout - drops ENTIRE channels"""
    def __init__(self, p: float = 0.25):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = torch.bernoulli(torch.ones(x.size(0), x.size(1), 1, device=x.device) * (1 - self.p))
        return x * mask / (1 - self.p)


class CausalConv1d(nn.Module):
    """Causal convolution with left-padding"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=0)
    
    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class LimitedLookaheadConv1d(nn.Module):
    """
    Convolution con lookahead limitato.
    
    Per applicazioni near-realtime: permette di guardare un po' nel futuro
    (es. 60 secondi) per migliorare le predizioni senza sacrificare troppo
    la latenza.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, max_lookahead=60):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.max_lookahead = max_lookahead
        
        # Calcola il lookahead effettivo di questa conv
        total_span = (kernel_size - 1) * dilation
        
        # Dividiamo: metà passato, metà futuro (o tutto passato se supera max_lookahead)
        ideal_future = total_span // 2
        self.future_padding = min(ideal_future, max_lookahead)
        self.past_padding = total_span - self.future_padding
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=0)
    
    def forward(self, x):
        # Pad: (left, right)
        x = F.pad(x, (self.past_padding, self.future_padding))
        return self.conv(x)


class MultiScaleCausalInception(nn.Module):
    """
    Multi-Scale Inception Block (Causal Version)
    
    Ispirato al paper STAFN: estrae features a scale diverse in parallelo.
    Usa kernel 1, 3, 5, 7 per catturare pattern a diverse granularità.
    
    Questo preserva l'informazione sulla magnitudine che viene persa
    attraverso i dilated convolution profondi.
    """
    def __init__(self, in_channels: int, out_channels: int, causal: bool = True):
        super().__init__()
        
        # Divide output channels tra le 4 branch
        branch_ch = out_channels // 4
        remainder = out_channels % 4
        
        self.causal = causal
        
        # Branch 1: kernel 1 (point-wise)
        self.branch1 = nn.Conv1d(in_channels, branch_ch, kernel_size=1)
        
        # Branch 2: kernel 3
        if causal:
            self.branch3 = CausalConv1d(in_channels, branch_ch, kernel_size=3)
        else:
            self.branch3 = nn.Conv1d(in_channels, branch_ch, kernel_size=3, padding=1)
        
        # Branch 3: kernel 5
        if causal:
            self.branch5 = CausalConv1d(in_channels, branch_ch, kernel_size=5)
        else:
            self.branch5 = nn.Conv1d(in_channels, branch_ch, kernel_size=5, padding=2)
        
        # Branch 4: kernel 7 (+ remainder per far tornare i channels)
        if causal:
            self.branch7 = CausalConv1d(in_channels, branch_ch + remainder, kernel_size=7)
        else:
            self.branch7 = nn.Conv1d(in_channels, branch_ch + remainder, kernel_size=7, padding=3)
        
        self.norm = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        # x: (B, C, T)
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        
        # Concat lungo channel dimension
        out = torch.cat([b1, b3, b5, b7], dim=1)
        out = self.norm(out)
        return F.gelu(out)


class CausalMultiHeadAttention(nn.Module):
    """
    Causal Multi-Head Attention with Flash Attention
    
    Usa F.scaled_dot_product_attention con is_causal=True per:
    - Flash Attention automatico su H200 (10x più veloce)
    - Memory efficient (O(T) invece di O(T²))
    - Tensor Core optimized
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer norm e feed-forward (transformer-style)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) - formato TCN
            is_causal: True per attention causale, False per bidirezionale
        
        Returns:
            out: (B, C, T) - stessa shape con attention applicata
        """
        # x: (B, C, T) → (B, T, C) per attention
        x = x.permute(0, 2, 1)
        B, T, C = x.shape
        
        # === Self-Attention con Flash Attention ===
        x_norm = self.norm1(x)
        
        # Project to Q, K, V
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Flash Attention (causale o bidirezionale)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal  # Controllato da parametro
        )
        
        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        attn_out = self.out_proj(attn_out)
        
        x = x + attn_out  # Residual
        
        # === Feed-Forward Network ===
        x = x + self.ffn(self.norm2(x))  # Residual
        
        # Torna a (B, C, T)
        return x.permute(0, 2, 1)


class ResidualBlockWithNorm(nn.Module):
    """TCN Residual Block con BatchNorm (migliorato)"""
    def __init__(self, channels: int, kernel_size: int = 2, 
                 dilation: int = 1, spatial_dropout: float = 0.25,
                 lookahead: int = 0):
        super().__init__()
        if lookahead > 0:
            self.conv = LimitedLookaheadConv1d(channels, channels, kernel_size, dilation, lookahead)
        else:
            self.conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = SpatialDropout1D(spatial_dropout)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return x + out


class DeepPowerHead(nn.Module):
    """
    Deep Power Regression Head
    
    3 layer MLP con GELU + BatchNorm per migliore regressione della potenza.
    Il paper STAFN usa 2 dense layers, noi ne usiamo 3 per capacità extra.
    """
    def __init__(self, in_features: int, hidden: int = 128, out_features: int = 1, dropout: float = 0.3):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden // 2, out_features)
        )
    
    def forward(self, x):
        return self.net(x)


class WaveNILMv6STAFN(nn.Module):
    """
    WaveNILM v6 - SOTA Causal Architecture for Real-Time NILM
    
    Architettura (192K parameters):
    1. Stem: 7×1 causal conv → estrae features iniziali
    2. TCN Backbone: 12 residual blocks con dilation 2^i (RF=4101)
    3. Skip Connections: stem → mid → final per multi-resolution
    4. Multi-Scale Inception: kernel 1,3,5,7 paralleli
    5. Flash Attention: is_causal=True per pattern temporali (SOLO per power head)
    6. Deep Power Head: 3-layer MLP per regressione accurata
    7. Gated Output: power = sigmoid(gate) × softplus(raw)
    
    DESIGN CHOICE CRITICA (paper-grade):
    - Gate usa features PRE-attention + media 5 min → decision boundary sharp
    - Power usa features POST-attention + ultimo timestep → smooth regression
    - Motivo: attention "spalma" la decision boundary, danneggia F1 del gate
    - Riferimenti: WaveNILM, TCAN usano gate da TCN puro
    
    Garanzie Real-Time:
    - Tutte le conv sono causali (left-padding)
    - Attention usa maschera triangolare
    - Output da ultimo timestep: h = x[:, -1, :]
    
    Input: (B, T=4096, F=7) - aggregate + 6 features temporali
    Output:
    - gate: (B, 1) probabilità ON/OFF [0, 1]
    - power_final: potenza gated = gate × softplus(power_raw)
    """
    def __init__(self,
                 n_features: int = 7,
                 n_appliances: int = 1,
                 hidden_channels: int = 64,
                 n_blocks: int = 10,
                 kernel_size: int = 2,
                 stem_kernel: int = 7,
                 spatial_dropout: float = 0.25,
                 head_dropout: float = 0.3,
                 use_psa: bool = True,
                 use_inception: bool = True,
                 lookahead: int = 0):  # 0=causale, >0=bidirezionale con N sec lookahead
        super().__init__()
        
        self.n_appliances = n_appliances
        self.use_psa = use_psa
        self.use_inception = use_inception
        self.lookahead = lookahead
        
        # === Stem ===
        if lookahead > 0:
            self.stem = nn.Sequential(
                LimitedLookaheadConv1d(n_features, hidden_channels, stem_kernel, max_lookahead=lookahead),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU()
            )
        else:
            self.stem = nn.Sequential(
                CausalConv1d(n_features, hidden_channels, stem_kernel),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU()
            )
        
        # === TCN Backbone (split in 2 parti per skip connection) ===
        n_blocks_first = n_blocks // 2
        n_blocks_second = n_blocks - n_blocks_first
        
        self.tcn_first = nn.ModuleList([
            ResidualBlockWithNorm(
                hidden_channels, 
                kernel_size=kernel_size,
                dilation=2**i,
                spatial_dropout=spatial_dropout,
                lookahead=lookahead
            )
            for i in range(n_blocks_first)
        ])
        
        self.tcn_second = nn.ModuleList([
            ResidualBlockWithNorm(
                hidden_channels, 
                kernel_size=kernel_size,
                dilation=2**(i + n_blocks_first),
                spatial_dropout=spatial_dropout,
                lookahead=lookahead
            )
            for i in range(n_blocks_second)
        ])
        
        # === Skip Connection Projections ===
        # Skip dalla stem
        self.skip_stem = nn.Conv1d(hidden_channels, hidden_channels // 2, 1)
        # Skip dal middle
        self.skip_mid = nn.Conv1d(hidden_channels, hidden_channels // 2, 1)
        
        # Fusion dei skip
        skip_total = hidden_channels + hidden_channels // 2 + hidden_channels // 2
        self.skip_fusion = nn.Sequential(
            nn.Conv1d(skip_total, hidden_channels, 1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU()
        )
        
        # === Multi-Scale Inception ===
        if use_inception:
            # Inception sempre causale per ora (lookahead handled by TCN)
            self.inception = MultiScaleCausalInception(hidden_channels, hidden_channels, causal=(lookahead == 0))
        
        # === Multi-Head Attention ===
        if use_psa:
            self.attention = CausalMultiHeadAttention(
                embed_dim=hidden_channels,
                num_heads=8,
                dropout=0.1
            )
            self.attention_is_causal = (lookahead == 0)  # Bidirezionale se lookahead > 0
        
        # === Output Heads ===
        head_input_dim = hidden_channels
        
        # Gate head: shared features + linear
        self.shared_gate = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.GELU(),
            nn.Dropout(head_dropout),
        )
        self.head_gate = nn.Linear(128, n_appliances)
        
        # Power head: deep MLP (key improvement from STAFN)
        self.power_head = DeepPowerHead(
            in_features=head_input_dim,
            hidden=128,
            out_features=n_appliances,
            dropout=head_dropout
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, target_timestep: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, F) input tensor
            target_timestep: indice del timestep target (default -1 = ultimo).
                            Con lookahead, usare -(lookahead+1) per il "presente".
        
        Returns:
            gate: (B, n_appliances) probabilità ON/OFF [0, 1]
            power_final: (B, n_appliances) potenza gated
        """
        # x: (B, T, F) → (B, F, T)
        x = x.transpose(1, 2)
        
        # === Stem ===
        x = self.stem(x)
        skip_stem = self.skip_stem(x)  # (B, C//2, T)
        
        # === TCN First Half ===
        for block in self.tcn_first:
            x = block(x)
        skip_mid = self.skip_mid(x)  # (B, C//2, T)
        
        # === TCN Second Half ===
        for block in self.tcn_second:
            x = block(x)
        
        # === Skip Fusion ===
        # Concatena: [stem_skip, mid_skip, final_output]
        x = torch.cat([skip_stem, skip_mid, x], dim=1)  # (B, C*2, T)
        x = self.skip_fusion(x)  # (B, C, T)
        
        # === Multi-Scale Inception ===
        if self.use_inception:
            x = self.inception(x)
        
        # === IMPORTANTE: salva features PRE-attention per il gate ===
        # L'attention "spalma" la decision boundary → danneggia F1 del gate
        # Ma aiuta la regressione → teniamola solo per power
        # Paper: WaveNILM, TCAN usano gate da TCN puro
        x_pre_attn = x
        
        # === Causal Multi-Head Attention (solo per power) ===
        if self.use_psa:
            x = self.attention(x, is_causal=self.attention_is_causal)
        
        # x: (B, C, T) → (B, T, C)
        x = x.transpose(1, 2)
        x_pre_attn = x_pre_attn.transpose(1, 2)
        
        T = x.size(1)
        
        # === Calcola indice target (supporta lookahead) ===
        # target_timestep=-1: ultimo timestep (causale, no lookahead)
        # target_timestep=-(lookahead+1): timestep "presente" con lookahead
        if target_timestep < 0:
            target_idx = T + target_timestep  # -1 → T-1, -(L+1) → T-L-1
        else:
            target_idx = target_timestep
        target_idx = max(0, min(target_idx, T - 1))  # Clamp to valid range
        
        # === Features per le due heads ===
        # Gate: media contesto PRE-attention → decision boundary sharp
        # Power: timestep target POST-attention → smooth regression
        gate_context = min(300, target_idx + 1)  # Fino al target (incluso)
        gate_start = target_idx - gate_context + 1
        h_gate = x_pre_attn[:, gate_start:target_idx + 1, :].mean(dim=1)  # (B, C)
        h_power = x[:, target_idx, :]  # (B, C) - timestep target
        
        # === Gate Head (usa features PRE-attention, media 5 min) ===
        h_gate_proj = self.shared_gate(h_gate)
        gate = torch.sigmoid(self.head_gate(h_gate_proj))  # (B, n_appliances)
        
        # === Power Head (usa features POST-attention, ultimo timestep) ===
        power_raw = self.power_head(h_power)  # (B, n_appliances)
        power_raw = F.softplus(power_raw)  # [0, ∞)
        
        # === Gated Output ===
        power_final = gate * power_raw
        
        return gate, power_final
    
    def get_receptive_field(self) -> int:
        """Calcola il receptive field totale"""
        n_blocks = len(self.tcn_first) + len(self.tcn_second)
        # RF = stem + sum(2^i * (k-1))
        rf = 6  # stem kernel 7 → 6 timesteps
        for i in range(n_blocks):
            rf += (2 ** i) * 1  # kernel_size=2, dilation=2^i → adds 2^i
        return rf


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test with n_blocks=12 for RF >= 4096
    model = WaveNILMv6STAFN(
        n_features=7,
        n_appliances=1,
        hidden_channels=64,
        n_blocks=12,  # RF = 6 + (2^12 - 1) = 4101 >= 4096 ✓
        use_psa=True,
        use_inception=True
    )
    
    rf = model.get_receptive_field()
    print(f"WaveNILM v6 STAFN Parameters: {count_parameters(model):,}")
    print(f"Receptive Field: {rf} timesteps (needs ≥4096)")
    assert rf >= 4096, f"RF {rf} < 4096! Increase n_blocks."
    
    # Test forward
    x = torch.randn(4, 4096, 7)  # (B, T, F) - 4096 timesteps input
    gate, power = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Gate shape: {gate.shape}, range: [{gate.min():.3f}, {gate.max():.3f}]")
    print(f"Power shape: {power.shape}, range: [{power.min():.3f}, {power.max():.3f}]")
    
    # Ablation configs
    print("\n=== Ablation Study Configs ===")
    
    configs = [
        ("Full v6 (Inception + Attention)", True, True),
        ("No Attention (Inception only)", True, False),
        ("No Inception (Attention only)", False, True),
        ("Baseline (no extras)", False, False),
    ]
    
    for name, use_inception, use_psa in configs:
        m = WaveNILMv6STAFN(use_inception=use_inception, use_psa=use_psa)
        print(f"{name}: {count_parameters(m):,} params")
