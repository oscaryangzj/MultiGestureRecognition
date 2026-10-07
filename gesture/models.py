import torch
from torch import nn
from torch.nn import functional as F


class StateMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, features):
        return self.layers(features)


class ActionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, features):
        # Right padding cannot affect earlier outputs of this causal LSTM.
        sequence, _hidden = self.lstm(features)
        return self.output(sequence)


class ActionCNNLSTM(nn.Module):
    """Meta discrete-gesture architecture adapted to camera feature sequences."""

    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 conv_channels, conv_kernel_size, dropout):
        super().__init__()
        self.left_context = conv_kernel_size - 1
        self.conv = nn.Conv1d(input_size, conv_channels, conv_kernel_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.post_conv_norm = nn.LayerNorm(conv_channels)
        # Explicit inter-layer dropout avoids the native MPS LSTM dropout cache bug.
        self.lstm = nn.ModuleList([
            nn.LSTM(conv_channels if layer == 0 else hidden_size, hidden_size, batch_first=True)
            for layer in range(num_layers)
        ])
        self.post_lstm_norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, features):
        # Left padding aligns every output with its newest available input frame.
        sequence = self.conv(F.pad(features.transpose(1, 2), (self.left_context, 0)))
        sequence = self.post_conv_norm(self.dropout(self.relu(sequence)).transpose(1, 2))
        for layer, lstm in enumerate(self.lstm):
            sequence, _hidden = lstm(sequence)
            if layer < len(self.lstm) - 1:
                sequence = self.dropout(sequence)
        return self.output(self.post_lstm_norm(sequence))
