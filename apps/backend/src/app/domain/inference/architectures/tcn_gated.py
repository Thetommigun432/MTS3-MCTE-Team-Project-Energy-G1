
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=self.padding, 
            dilation=dilation
        )
        
    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            return x[:, :, :-self.padding]
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
        
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        
        # Gated Activation: produces 2 * channels (Filter + Gate)
        self.conv = CausalConv1d(
            in_channels, 
            out_channels * 2, 
            kernel_size, 
            dilation=dilation
        )
        
        self.norm = nn.BatchNorm1d(out_channels * 2)
        # SE block applied AFTER gated split to out_channels (not out_channels*2)
        self.se = SEBlock(out_channels)
        
        self.residual_conv = nn.Conv1d(out_channels, out_channels, 1)
        self.skip_conv = nn.Conv1d(out_channels, out_channels, 1)
        
    def forward(self, x):
        # x: (B, C, T)
        
        # 1. Causal Conv (Dilated)
        out = self.conv(x)
        out = self.norm(out)
        
        # 2. Split for Gated Activation
        filter_out, gate_out = out.chunk(2, dim=1)
        z = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        
        # 3. SE block on gated output
        z = self.se(z)
        
        # 4. Projections
        res = self.residual_conv(z)
        skip = self.skip_conv(z)
        
        return (x + res) * 0.6, skip

class TCN_Gated(nn.Module):
    """
    TCN_Gated Architecture - Gated TCN for NILM.
    Features:
    - 2-Stack Residual Network
    - Squeeze-and-Excitation Blocks
    - Gated Dilated Causal Convolutions
    - Multi-Task Learning (Power + Probability)
    """
    def __init__(
        self,
        n_input_features: int = 7,
        hidden_channels: int = 64, # Default from checkpoint inspection
        n_blocks: int = 9,         # Default from checkpoint inspection
        n_stacks: int = 1,         # Inspection didn't distinguish stacks clearly, but n_blocks=9 usually implies single stack or total blocks
        kernel_size: int = 3,
        use_mtl: bool = True,
        **kwargs 
    ):
        super().__init__()
        
        self.n_input_features = n_input_features
        self.hidden_channels = hidden_channels
        self.n_blocks = n_blocks
        self.use_mtl = use_mtl
        
        # Initial convolution
        # Matches key: input_conv.0.conv.weight
        self.input_conv = nn.Sequential(
             CausalConv1d(
                 n_input_features, 
                 hidden_channels, 
                 kernel_size=1
             )
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        # Create blocks for all stacks
        total_blocks = n_blocks * n_stacks
        
        for i in range(total_blocks):
            dilation = 2 ** (i % n_blocks) 
            
            self.blocks.append(
                ResidualBlock(
                    hidden_channels, 
                    hidden_channels, 
                    kernel_size, 
                    dilation
                )
            )
            
        # Output layers
        self.skip_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        
        # MTL Heads
        # Matches keys: "regression_head.0.weight", "classification_head.0.weight"...
        # Assuming simple convolution if keys are just weight/bias, or Sequential if numbered.
        # Logs showed "regression_head.4.weight", implies Sequential.
        # Standard: Conv(128->64) -> ReLU -> Conv(64->1)
        
        self.regression_head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels // 2, 1), # Guessing layers based on .4 index
            nn.ReLU(),
            nn.Conv1d(hidden_channels // 2, 1, 1)
        )
        
        self.classification_head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels // 2, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels // 2, 1, 1)
        )
        
    def forward(self, x):
        # Permute (B, T, C) -> (B, C, T)
        if x.shape[-1] == self.n_input_features and x.shape[1] != self.n_input_features:
            x = x.transpose(1, 2)
            
        x = self.input_conv(x)
        
        # Accumulate skips
        skips = 0
        for block in self.blocks:
            x, skip = block(x)
            skips = skips + skip
            
        # Output processing
        x = F.relu(skips)
        x = self.skip_conv(x) 
        x = F.relu(x)
        
        power = self.regression_head(x)
        prob = self.classification_head(x)
        
        return power.transpose(1, 2), torch.sigmoid(prob.transpose(1, 2))
