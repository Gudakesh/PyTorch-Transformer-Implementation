import torch
from layer_normalization_block import LayerNormalization

class ResidualConnection(torch.nn.Module):
   def __init__(self, size: int, dropout: float):
      super().__init__()
      self.norm = LayerNormalization(features = size)
      self.dropout = torch.nn.Dropout(p = dropout)
   
   def forward(self, x: torch.tensor, subLayer: torch.nn.Module) -> torch.tensor:
      """s
      takes 2 inputs : 
      1) Previous layer (subLayer)
      2) input to previous layer (x)

      Output :
      takes input & output of subLayer, adds them, then does the normalization, that becomes the output
      basically : LayerNorm(x + subLayer(x)) is the output
      """
      return self.norm(x + self.dropout(subLayer(x)))