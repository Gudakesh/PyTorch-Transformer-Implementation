import torch

class FeedForwardBlock(torch.nn.Module):
   def __init__(self, d_model: int, d_ff: int, dropout: float):
      super().__init__()
      self.layer_1 = torch.nn.Linear(in_features = d_model, out_features = d_ff)
      self.dropout = torch.nn.Dropout(dropout=dropout)
      self.layer_2 = torch.nn.Linear(in_features = d_ff, out_features = d_model)
   
   def forward(self, x: torch.tensor) -> torch.tensor:
      # x: (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
      # L2 -> dropout -> ReLU -> L1
      return self.layer_2(self.dropout(torch.relu(self.layer_1(x))))