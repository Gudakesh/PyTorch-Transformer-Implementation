import torch

from blocks.multi_head_attention_block import MultiHeadAttentionBlock
from blocks.feed_forward_block import FeedForwardBlock
from blocks.add_and_norm_block import ResidualConnection

class DecoderLayer(torch.nn.Module):
   def __init__(
      self,
      features: int,
      self_attention: MultiHeadAttentionBlock,
      cross_attention: MultiHeadAttentionBlock,
      feed_forward: FeedForwardBlock,
      dropout: float
   ):
      self.size = features
      self.self_attention_block = self_attention
      self.cross_attention_block = cross_attention
      self.feed_forward_block = feed_forward
      self.add_and_norm_block_1 = ResidualConnection(size = self.size, dropout = dropout)
      self.add_and_norm_block_2 = ResidualConnection(size = self.size, dropout = dropout)
      self.add_and_norm_block_3 = ResidualConnection(size = self.size, dropout = dropout)
   
   def forward(self, x: torch.tensor, encoder_output: torch.tensor, encoder_mask: torch.tensor, decoder_mask: torch.tensor):
      # need to use lambda function to redirect callable inside ResidualConnection class, as it takes only single argument, and we need to use multiple input arguments for our function.
      # so using lambda to map SubLayer(x) -> SubLayer(x, x, x, mask)
      x = self.add_and_norm_block_1(x, lambda x: self.self_attention_block(q = x, k = x, v = x, mask = decoder_mask))
      x = self.add_and_norm_block_2(x, lambda x: self.self_attention_block(q = x, k = encoder_output, v = encoder_output, mask = encoder_mask))
      # no need to use lambda for redirection in FeedForward sublayer, as it only takes single input
      x = self.add_and_norm_block_3(x, self.feed_forward_block)
      return x