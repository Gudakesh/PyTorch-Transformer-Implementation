import torch

from layers.embeddings_to_tokens_layer import ProjectionLayer

from blocks.embedding_block import Embeddings
from blocks.positional_encoding_block import PositionalEncoding
from blocks.layer_normalization_block import LayerNormalization

class Encoder(torch.nn.Module):
   def __init__(self, layers: torch.nn.ModuleList):
      super().__init__()
      self.layers = layers
      self.norm = LayerNormalization(features = self.layers[0].size)
   
   def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
      for layer in self.layers:
         x = layer(x, encoder_mask = mask)
      return self.norm(x)

class Decoder(torch.nn.Moudle):
   def __init__(self, layers: torch.nn.ModuleList):
      super().__init__()
      self.layers = layers
      self.norm = LayerNormalization(features = self.layers[0].size)
   
   def forward(self, x: torch.tensor, encoder_output: torch.tensor, encoder_mask: torch.tensor, decoder_mask: torch.tensor) -> torch.tensor:
      for layer in self.layers:
         x = layer(x, encoder_output, encoder_mask, decoder_mask)
      return self.norm(x)

class Transformer(torch.nn.Module):
   def __init__(
      self, 
      encoder: Encoder,
      decoder: Decoder,
      input_embedding: Embeddings,
      output_embedding: Embeddings,
      input_positional_encoding: PositionalEncoding,
      output_positional_encoding: PositionalEncoding,
      embeddings_to_tokens: ProjectionLayer
   ):
      super().__init__()
      self.encoder = encoder
      self.decoder = decoder
      self.input_embedding = input_embedding
      self.output_embedding = output_embedding
      self.input_pos_encoding = input_positional_encoding
      self.output_pos_encoding = output_positional_encoding
      self.embed_to_tokens_projection_layer = embeddings_to_tokens
   
   def encode(self, x: torch.tensor, encoder_mask: torch.tensor) -> torch.tensor:
      x = self.input_embedding(x)
      x = self.input_pos_encoding(x)
      return self.encoder(x, mask = encoder_mask)
   
   def decode(self, x: torch.tensor, encoder_output: torch.tensor, encoder_mask: torch.tensor, decoder_mask: torch.tensor) -> torch.tensor:
      x = self.output_embedding(x)
      x = self.output_pos_encoding(x)
      return self.decoder(x, encoder_output, encoder_mask, decoder_mask)

   def project_embed_to_token(self, x: torch.tensor) -> torch.tensor:
      # (batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
      return self.embed_to_tokens_projection_layer(x)

   def forward(self, input_sequence: torch.tensor, output_sequence: torch.tensor, input_mask: torch.tensor, output_mask: torch.tensor) -> torch.tensor:
      encoder_output = self.encode(x = input_sequence, encoder_mask = input_mask)
      decoder_output = self.decode(x = output_sequence, encoder_output = encoder_output, encoder_mask = input_mask, decoder_mask = output_mask)
      return self.project_embed_to_token(decoder_output)