
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

class WaveNILM_v3(nn.Module):
    """
    WaveNILM v3 Architecture - Reconstructed from SOTA Checkpoints.
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
        self.first_conv = nn.Conv1d(
            n_input_features, 
            hidden_channels, 
            kernel_size=1
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        # Assumed flat list of blocks based on "blocks.0", "blocks.1"...
        for i in range(n_blocks):
            dilation = 2 ** (i % 9) # Reset dilation every 9 blocks if multiple stacks? 
            # Or just standard wavenet exp growth.
            # Checkpoint had block 0..8, likely 9 blocks total.
            
            self.blocks.append(
                ResidualBlock(
                    hidden_channels, 
                    hidden_channels, 
                    kernel_size, 
                    dilation
                )
            )
            
        # Output layers
        self.post_conv = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=1)
        # Checkpoint inspection needed for post_conv size. 
        # Usually it's relu -> 1x1 -> relu -> 1x1.
        # If "post_conv" key exists. 
        
        # MTL Heads
        self.power_out = nn.Conv1d(hidden_channels * 2, 1, kernel_size=1)
        self.prob_out = nn.Conv1d(hidden_channels * 2, 1, kernel_size=1)
        
    def forward(self, x):
        # Permute (B, T, C) -> (B, C, T)
        if x.shape[-1] == self.n_input_features and x.shape[1] != self.n_input_features:
            x = x.transpose(1, 2)
            
        x = self.first_conv(x)
        
        # Accumulate skips
        skips = 0
        for block in self.blocks:
            x, skip = block(x)
            skips = skips + skip
            
        # Output processing
        # Typically: ReLU -> 1x1 -> ReLU -> 1x1
        # Reconstruct based on standard patterns if keys strictly match "post_conv"
        
        x = F.relu(skips)
        # Note: If post_conv weight is [128, 64, 1], then output is 128.
        # This matches hidden*2 intermediate expansion often used.
        x = self.post_conv(x) 
        x = F.relu(x)
        
        power = self.power_out(x)
        prob = self.prob_out(x)
        
        return power.transpose(1, 2), torch.sigmoid(prob.transpose(1, 2))
