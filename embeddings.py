import torch
import math

class Embeddings(torch.nn.Module):
   def __init__(self, d_model: int, vocab_size: int):
      super().__init__()
      self.d_model = d_model
      self.vocab_size = vocab_size
      self.embed = torch.nn.Embedding(
         num_embeddings = self.vocab_size,
         embedding_dim = self.d_model
      )
   
   def forward(self, x: torch.tensor) -> torch.tensor:
      return self.embed(x) * math.sqrt(self.d_model)