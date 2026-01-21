"""
NILM Energy Monitor - Model Architectures
PyTorch model definitions for appliance power disaggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNSeq2Seq(nn.Module):
    """CNN Encoder-Decoder for NILM power disaggregation"""

    def __init__(self, input_channels=7, hidden_channels=48, num_layers=3):
        super(CNNSeq2Seq, self).__init__()

        # Build encoder layers
        encoder_layers = []
        in_ch = input_channels
        for i in range(num_layers):
            out_ch = hidden_channels * (2 ** i)
            encoder_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.25)
            ])
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_layers)

        self.bottleneck_ch = hidden_channels * (2 ** (num_layers - 1))

        # Build decoder layers
        decoder_layers = []
        in_ch = self.bottleneck_ch
        for i in range(num_layers - 1, -1, -1):
            out_ch = hidden_channels * (2 ** i) if i > 0 else hidden_channels
            decoder_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            in_ch = out_ch
        self.decoder = nn.Sequential(*decoder_layers)

        self.output_layer = nn.Sequential(
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output = self.output_layer(decoded)
        return output.transpose(1, 2)  # (batch, seq_len, 1)


class UNet1D(nn.Module):
    """U-Net 1D for NILM with skip connections"""

    def __init__(self, input_channels=7, base_channels=24):
        super(UNet1D, self).__init__()

        # Encoder
        self.enc1 = self._conv_block(input_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16)

        # Decoder (with skip connections)
        self.dec4 = self._conv_block(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.dec3 = self._conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = self._conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = self._conv_block(base_channels * 2 + base_channels, base_channels)

        self.output = nn.Sequential(
            nn.Conv1d(base_channels, 1, kernel_size=1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool1d(2)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25)
        )

    def _upsample_and_concat(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2], mode='linear', align_corners=True)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        original_len = x.shape[1]
        x = x.transpose(1, 2)  # (batch, features, seq_len)

        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder path with skip connections
        d4 = self.dec4(self._upsample_and_concat(b, e4))
        d3 = self.dec3(self._upsample_and_concat(d4, e3))
        d2 = self.dec2(self._upsample_and_concat(d3, e2))
        d1 = self.dec1(self._upsample_and_concat(d2, e1))

        out = self.output(d1)

        # Ensure output matches input length
        if out.shape[2] != original_len:
            out = F.interpolate(out, size=original_len, mode='linear', align_corners=True)

        return out.transpose(1, 2)  # (batch, seq_len, 1)


# Model registry mapping
MODEL_CLASSES = {
    'cnn': CNNSeq2Seq,
    'cnnseq2seq': CNNSeq2Seq,
    'unet': UNet1D,
    'unet1d': UNet1D,
}


def get_model_class(model_type: str):
    """
    Get model class by type name.

    Args:
        model_type: Model type identifier (case-insensitive)

    Returns:
        Model class

    Raises:
        ValueError: If model type is not recognized
    """
    model_type_lower = model_type.lower()
    if model_type_lower not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available types: {list(MODEL_CLASSES.keys())}"
        )
    return MODEL_CLASSES[model_type_lower]


def create_model(model_type: str, input_channels: int = 7, **kwargs):
    """
    Create a model instance by type.

    Args:
        model_type: Model type identifier
        input_channels: Number of input features
        **kwargs: Additional model-specific parameters

    Returns:
        Instantiated model
    """
    model_class = get_model_class(model_type)
    return model_class(input_channels=input_channels, **kwargs)
