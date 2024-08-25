import torch

class LayerNormalization(torch.nn.Module):
   def __init__(self, features: int, eps: float = 10**-6):
      super().__init__()
      self.eps = eps
      # multiplicative term for doing normalization
      self.alpha = torch.nn.Parameter(torch.ones(features))
      # additive term for doing normalization
      self.beta = torch.nn.Parameter(torch.zeros(features))
   
   def forward(self, x: torch.tensor) -> torch.tensor:
      mean = torch.mean(input = x, dim = -1, keepdim = True)
      std_dev = torch.std(input = x, dim = -1, keepdim = True)

      return self.alpha * ( (x-mean)/(std_dev + self.eps) ) + self.beta