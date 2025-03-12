# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC

class TDSAttnBlock(nn.Module):
    def __init__(self, channels: int, width: int, num_heads: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.C = self.channels * self.width

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.C, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # T_in, N, C = inputs.shape  # TNC

        # TNC -> NTC
        x = inputs.permute(1, 0, 2)
        x = self.encoder(x)
        # NCT -> TNC
        x = x.permute(1, 0, 2) 
        return x

class TDSCNNLSTMBlock(nn.Module):
    """A 2D temporal convolution block with LSTM layer as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619), with an additional LSTM layer for sequence modeling.

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
        hidden_size (int, optional): The number of features in the hidden state of LSTM.
            Defaults to None, which means it's equal to channels.
    """

    def __init__(self, channels: int, width: int, kernel_width: int, hidden_size: int = None) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.hidden_size = hidden_size if hidden_size is not None else channels

        # Temporal convolution part (same as TDSConv2dBlock)
        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        
        # LSTM layer for sequence modeling
        # Input shape after conv: (T_out, N, channels * width)
        self.lstm = nn.LSTM(
            input_size=channels * width,
            hidden_size=self.hidden_size,
            batch_first=False,  # We use (T, N, C) format
            bidirectional=True  # Bidirectional for better context
        )
        
        # Project bidirectional LSTM output back to original feature size
        self.proj = nn.Linear(self.hidden_size * 2, channels * width)
        
        # Layer normalization for the final output
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC (Time, Batch, Features)

        # 1. Apply temporal convolution (same as TDSConv2dBlock)
        # TNC -> NCT -> NcwT (Batch, Channels, Width, Time)
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Keep track of the new time dimension after convolution
        T_out = x.shape[0]
        
        # 2. Apply bidirectional LSTM
        lstm_out, _ = self.lstm(x)  # TNC -> TNC (but with hidden_size*2 features)
        
        # 3. Project back to original feature size
        lstm_out = self.proj(lstm_out)  # TNC -> TNC (original size)
        
        # 4. Skip connection (add the output of temporal convolution)
        x = lstm_out + x
        
        # 5. Skip connection with original input (after accounting for time dimension change)
        x = x + inputs[-T_out:]
        
        # 6. Layer normalization
        return self.layer_norm(x)  # TNC

class TDSLSTMBlock(nn.Module):
    """A pure LSTM block for time-depth separable architectures. This block uses only
    an LSTM for sequence modeling without the convolutional component, intended as a baseline.
    Compatible with the TDS architecture's channel/width paradigm.

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        hidden_size (int, optional): The number of features in the hidden state of LSTM.
            Defaults to None, which means it's equal to channels.
    """

    def __init__(self, channels: int, width: int, hidden_size: int = None) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.num_features = channels * width
        self.hidden_size = hidden_size if hidden_size is not None else channels
        
        # LSTM layer for sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.hidden_size,
            batch_first=False,  # We use (T, N, C) format
            bidirectional=True  # Bidirectional for better context
        )
        
        # Project bidirectional LSTM output back to original feature size
        self.proj = nn.Linear(self.hidden_size * 2, self.num_features)
        
        # Layer normalization for the final output
        self.layer_norm = nn.LayerNorm(self.num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC
        assert C == self.num_features, f"Expected {self.num_features} features, got {C}"
        
        # Apply bidirectional LSTM directly to input
        lstm_out, _ = self.lstm(inputs)  # TNC -> TNC (but with hidden_size*2 features)
        
        # Project back to original feature size
        x = self.proj(lstm_out)  # TNC -> TNC (original size)
        
        # Skip connection
        x = x + inputs
        
        # Layer normalization
        return self.layer_norm(x)  # TNC




class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)
    
    
class TDSAttnEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
       
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_attn_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_attn_blocks.extend(
                [
                    TDSAttnBlock(channels, num_features // channels),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_attn_blocks = nn.Sequential(*tds_attn_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_attn_blocks(inputs)  # (T, N, num_features)
    
class TDSLSTMEncoder(nn.Module):
    """A time depth-separable LSTM encoder composing a sequence of `TDSLSTMBlock`
    following the structure of TDSConvEncoder.

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per block. Used to determine width for each block.
        hidden_size (int, optional): The number of features in the hidden state of LSTM.
            Defaults to None, which means it's equal to the number of features for each block.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        hidden_size: int = None,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_lstm_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            
            # For each block, create a TDSLSTMBlock
            width = num_features // channels
            
            tds_lstm_blocks.extend(
                [
                    TDSLSTMBlock(channels, width, hidden_size),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        
        self.tds_lstm_blocks = nn.Sequential(*tds_lstm_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_lstm_blocks(inputs)  # (T, N, num_features)


class TDSCNNLSTMEncoder(nn.Module):
    """A time depth-separable CNN+LSTM encoder composing a sequence of `TDSCNNLSTMBlock`.
    This encoder combines convolutional processing with LSTM for sequence modeling.

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSCNNLSTMBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
        hidden_size (int, optional): The number of features in the hidden state of LSTM.
            Defaults to None, which will use the number of channels as hidden size.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
        hidden_size: int = None,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0, "Must specify at least one block"
        cnn_lstm_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            width = num_features // channels
            cnn_lstm_blocks.extend(
                [
                    TDSCNNLSTMBlock(channels, width, kernel_width, hidden_size),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.cnn_lstm_blocks = nn.Sequential(*cnn_lstm_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.cnn_lstm_blocks(inputs)  # (T, N, num_features)