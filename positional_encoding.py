import torch
import math

class PositionalEncoding(torch.nn.Module):
   def __init__(self, d_model: int, max_seq_len: int, dropout: float):
      super().__init__()
      self.d_model = d_model
      self.seq_len = max_seq_len
      
      self.dropout = torch.nn.Dropout(p = dropout)
      
      # Shape : (Seq_len, d_model)
      pe_matrix = torch.zeros(self.seq_len, self.d_model)

      # shape : (Seq_len, 1)
      position = torch.arange(0, self.seq_len, dtype=torch.float32).unsqueeze(dim=1)

      # formula : pos / ( 10000^(2i/d_model) )
      # 10000^(2i/d_model) can be rewritten as : e^x where x = log-to-base-e(10000)*(2i/d_model)
      denomentaor_term = torch.exp( torch.arange(start=0, end=self.d_model, step=2).float() * (-math.log(10000.0) / self.d_model) )
      # shape : (d_model/2)

      # sine to even positions
      pe_matrix[:, 0::2] = torch.sin(position * denomentaor_term)

      # cosine to odd positions
      pe_matrix[:, 1::2] = torch.cos(position * denomentaor_term)

      # adding batch dimension to pe matrix : new shape (1, Seq_len, d_model)
      pe_matrix = pe_matrix.unsqueeze(dim=0)

      # adding the pe matrix to this module's register buffer
      # for putting a tensor in model's state_dict even though it's not a trainable parameter, so that it gets saved alongside model
      self.register_buffer(name = "pe_matrix", tensor = pe_matrix)
   
   def forward(self, x: torch.tensor) -> torch.tensor:
      # telling the model that it's not a learned parameter
      x = x + (self.pe_matrix[:, :x.shape[1], :]).requires_grad_(False)
      return self.dropout(x)