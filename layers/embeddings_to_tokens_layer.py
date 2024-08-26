import torch

class ProjectionLayer(torch.nn.Module):
   def __init__(self, d_model: int, vocab_size: int):
      super().__init__()
      self.project = torch.nn.Linear(in_features = d_model, out_features = vocab_size)
   
   def forward(self, x: torch.tensor) -> torch.tensor:
      # (batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
      # appying log_softmax for better numerical stability
      return torch.nn.functional.log_softmax(self.project(x), dim = -1)