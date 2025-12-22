"""
Neural Network Models for Rough Volatility Transfer Learning

This module implements the transfer learning architecture for volatility forecasting:
    1. RoughVolatilityEncoder: Learns transferable volatility features (LSTM + Attention)
    2. RoughVolatilityPredictor: Task-specific prediction head (MLP)
    3. TransferRoughVolModel: Complete model combining encoder + predictor

Architecture Design:
    - Encoder captures universal "rough volatility" patterns
    - Attention mechanism identifies important time steps
    - Bottleneck layer (32-dim) forces learning of compact representations
    - Predictor adapts to specific asset characteristics

Usage:
    model = TransferRoughVolModel(hidden_dim=64, encoding_dim=32)
    model.freeze_encoder()  # For fine-tuning
    predictions = model(X)

Author: Ronit Dhansoia
Date: 22nd December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    Allows the model to attend to different positions in the sequence,
    helping identify which past volatility observations are most relevant
    for prediction.

    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        scale: Scaling factor for attention scores
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize Multi-Head Attention.

        Args:
            hidden_dim: Input/output dimension
            num_heads: Number of attention heads (default: 4)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention: (batch, heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Reshape back: (batch, seq_len, hidden_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Output projection
        output = self.out_proj(context)

        return output


class RoughVolatilityEncoder(nn.Module):
    """
    Encoder for learning transferable rough volatility features.

    Architecture:
        1. 3-layer LSTM to capture temporal dependencies
        2. Multi-head attention over LSTM outputs
        3. Bottleneck layer to 32-dim encoding

    This component learns universal volatility patterns that transfer
    across different asset classes.

    Attributes:
        lstm: 3-layer bidirectional LSTM
        attention: Multi-head self-attention
        bottleneck: Linear layer to encoding dimension
    """

    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 64,
                 encoding_dim: int = 32,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize the encoder.

        Args:
            input_dim: Input feature dimension (default: 1 for univariate)
            hidden_dim: LSTM hidden dimension (default: 64)
            encoding_dim: Final encoding dimension (default: 32)
            num_layers: Number of LSTM layers (default: 3)
            num_heads: Number of attention heads (default: 4)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim

        # 3-layer LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Layer normalization after LSTM
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)

        # Bottleneck to encoding dimension
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to fixed-size representation.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Encoding tensor of shape (batch, encoding_dim)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)

        # Self-attention over LSTM outputs
        attn_out = self.attention(lstm_out)  # (batch, seq_len, hidden_dim)

        # Take the last time step (or could use mean pooling)
        final_hidden = attn_out[:, -1, :]  # (batch, hidden_dim)

        # Bottleneck to encoding
        encoding = self.bottleneck(final_hidden)  # (batch, encoding_dim)

        return encoding


class RoughVolatilityPredictor(nn.Module):
    """
    Prediction head for volatility forecasting.

    Architecture:
        MLP: encoding_dim → 64 → 32 → forecast_horizon

    This component is fine-tuned for each target asset while
    keeping the encoder frozen (or jointly trained).

    Attributes:
        mlp: Multi-layer perceptron for prediction
    """

    def __init__(self,
                 encoding_dim: int = 32,
                 forecast_horizon: int = 1,
                 dropout: float = 0.1):
        """
        Initialize the predictor.

        Args:
            encoding_dim: Input encoding dimension (default: 32)
            forecast_horizon: Number of steps to predict (default: 1)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, forecast_horizon)
        )

    def forward(self, encoding: torch.Tensor) -> torch.Tensor:
        """
        Predict future volatility from encoding.

        Args:
            encoding: Encoded representation of shape (batch, encoding_dim)

        Returns:
            Predictions of shape (batch, forecast_horizon) or (batch,) if horizon=1
        """
        output = self.mlp(encoding)

        # Squeeze if single-step prediction
        if output.shape[-1] == 1:
            output = output.squeeze(-1)

        return output


class TransferRoughVolModel(nn.Module):
    """
    Complete Transfer Learning Model for Rough Volatility Forecasting.

    Combines:
        - RoughVolatilityEncoder: Learns transferable features
        - RoughVolatilityPredictor: Task-specific predictions

    Training Protocol:
        1. Pre-train entire model on source domain (S&P 500)
        2. Freeze encoder, fine-tune predictor on target domain (Bitcoin)
        3. Optionally unfreeze encoder for joint fine-tuning

    Attributes:
        encoder: RoughVolatilityEncoder instance
        predictor: RoughVolatilityPredictor instance
    """

    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 64,
                 encoding_dim: int = 32,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 forecast_horizon: int = 1,
                 dropout: float = 0.1):
        """
        Initialize the transfer learning model.

        Args:
            input_dim: Input feature dimension (default: 1)
            hidden_dim: LSTM hidden dimension (default: 64)
            encoding_dim: Bottleneck encoding dimension (default: 32)
            num_layers: Number of LSTM layers (default: 3)
            num_heads: Number of attention heads (default: 4)
            forecast_horizon: Number of steps to predict (default: 1)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()

        self.encoder = RoughVolatilityEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            encoding_dim=encoding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        self.predictor = RoughVolatilityPredictor(
            encoding_dim=encoding_dim,
            forecast_horizon=forecast_horizon,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and predictor.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Predictions of shape (batch,) or (batch, forecast_horizon)
        """
        encoding = self.encoder(x)
        predictions = self.predictor(encoding)
        return predictions

    def get_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the encoded representation (useful for analysis).

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Encoding of shape (batch, encoding_dim)
        """
        return self.encoder(x)

    def freeze_encoder(self):
        """Freeze encoder parameters for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("✓ Encoder frozen")

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for joint training."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("✓ Encoder unfrozen")

    def get_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# TESTING / EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MODEL ARCHITECTURE")
    print("=" * 60)

    # Create model
    model = TransferRoughVolModel(
        input_dim=1,
        hidden_dim=64,
        encoding_dim=32,
        num_layers=3,
        num_heads=4,
        forecast_horizon=1,
        dropout=0.1
    )

    print("\n📊 Model Architecture:")
    print(model)

    print(f"\n📈 Parameter Count:")
    print(f"  Total parameters: {model.get_total_params():,}")
    print(f"  Trainable parameters: {model.get_trainable_params():,}")

    # Test forward pass
    print("\n🔍 Testing Forward Pass:")
    batch_size = 32
    seq_len = 20
    X = torch.randn(batch_size, seq_len, 1)

    with torch.no_grad():
        # Full forward pass
        predictions = model(X)
        print(f"  Input shape: {X.shape}")
        print(f"  Output shape: {predictions.shape}")

        # Get encodings
        encodings = model.get_encoding(X)
        print(f"  Encoding shape: {encodings.shape}")

    # Test freezing
    print("\n❄️ Testing Encoder Freezing:")
    model.freeze_encoder()
    print(f"  Trainable after freeze: {model.get_trainable_params():,}")

    model.unfreeze_encoder()
    print(f"  Trainable after unfreeze: {model.get_trainable_params():,}")

    # Test with different sequence lengths
    print("\n📏 Testing Different Sequence Lengths:")
    for seq_len in [10, 20, 50, 100]:
        X = torch.randn(16, seq_len, 1)
        with torch.no_grad():
            out = model(X)
        print(f"  seq_len={seq_len}: input {X.shape} → output {out.shape}")

    print("\n✅ Model tests passed!")
