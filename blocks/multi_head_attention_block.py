import torch
import math

def scalar_dot_product_attention(
   query: torch.tensor, 
   key: torch.tensor,
   value: torch.tensor,
   mask: torch.tensor = None,
   dropout: torch.nn.Dropout = None
):
   # query, key, value : Shape :: (batch_size, num_heads, seq_len, d_k)
   # mask : shape :: (batch_size, num_heads, seq_len, seq_len)
   d_k = query.shape[-1]
   
   # attention = softmax[ (Q.K_transpose)/root(d_k) ] * Value
   # (batch_size, num_heads, seq_len, d_k) * (batch_size, num_heads, d_k, seq_len) --> (batch_size, num_heads, seq_len, seq_len)
   attention_scores = torch.matmul(query, torch.transpose(key, -2, -1)) / math.sqrt(d_k)

   # apply mask(if given) before doing softmax (used in inference in decoder, and for hiding padding words in an input sequence in encoder)
   if mask is not None:
      attention_scores = attention_scores.masked_fill_(mask==0, -1e9)
   
   # apply softmax
   attention_scores = torch.softmax(attention_scores, dim = -1)

   if dropout is not None:
      attention_scores = dropout(attention_scores)
   
   # multiplying with value to get output vector
   # (batch_size, num_heads, seq_len, seq_len) * (batch_size, num_heads, seq_len, d_k) --> (batch_size, num_heads, seq_len, d_k)
   v = torch.matmul(attention_scores, value)

   return v, attention_scores

class MultiHeadAttentionBlock(torch.nn.Module):
   def __init__(self, d_model: int, num_heads: int, dropout: float):
      super().__init__()
      
      self.d_model = d_model
      self.num_heads = num_heads

      assert self.d_model % self.num_heads == 0, "D_model is not divisible by num_heads"

      self.d_k = self.d_model // self.num_heads
      
      # W_q, W_k, W_v, W_o : (d_model, d_model)
      self.w_q = torch.nn.Linear(in_features = self.d_model, out_features = self.d_model)
      self.w_k = torch.nn.Linear(in_features = self.d_model, out_features = self.d_model)
      self.w_v = torch.nn.Linear(in_features = self.d_model, out_features = self.d_model)
      self.w_o = torch.nn.Linear(in_features = self.d_model, out_featuers = self.d_model)

      self.dropout = torch.nn.Dropout(p = dropout)
   
   def forward(self, q: torch.tensor, k: torch.tensor, v: torch.tensor, mask: torch.tensor) -> torch.tensor:
      # q,k,v shape : (batch_size, seq_len, d_model)

      batch_size = q.shape[0]
      seq_len = q.shape[1]

      query = self.w_q(q)
      key = self.w_k(k)
      value = self.w_v(v)

      # query,key,value : shape : (batch_size, seq_len, d_model)
      # since all heads are concatenated, split it into (batch_size, seq_len, num_heads, d_k) & then reshape it so that each head sees embeddings for complete sequence
      # (batch_size, seq_len, d_model) --> (batch_size, seq_len, num_heads, d_k) --> (batch_size, num_heads, seq_len, d_k)
      query = torch.transpose(query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k), 2, 1)
      key = torch.transpose(key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k), 2, 1)
      value = torch.transpose(value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k), 2, 1)

      x, self.scores = scalar_dot_product_attention(query, key, value, mask, self.dropout)

      # swap the num_heads & seq_len dims again & then recombine all the heads
      # (batch_size, num_heads, seq_len, d_k) --> (batch_size, seq_len, num_heads, d_k) --> flattened
      x = torch.transpose(x, 1, 2).contiguous()

      # make it in shape (batch_size, seq_len, d_model) where d_model = num_heads * d_k
      x = x.view(batch_size, seq_len, self.num_heads * self.d_k)

      # shape : (batch_size, seq_len, d_model)
      return self.w_o(x)