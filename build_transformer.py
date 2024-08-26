from blocks.embedding_block import Embeddings
from blocks.feed_forward_block import FeedForwardBlock
from blocks.positional_encoding_block import PositionalEncoding
from blocks.multi_head_attention_block import MultiHeadAttentionBlock

from layers.encoder_layer import EncoderLayer
from layers.decoder_layer import DecoderLayer
from layers.embeddings_to_tokens_layer import ProjectionLayer

from models.transformer import Encoder
from models.transformer import Decoder
from models.transformer import Transformer

def build_transformer(
   input_vocab_size: int, 
   output_vocab_size: int, 
   input_seq_len: int, 
   output_seq_len: int, 
   d_model: int = 512,
   d_ff: int = 2048,
   num_encoding_layers: int = 6,
   num_decoding_layers: int = 6,
   num_attention_heads: int = 4,
   dropout: float = 0.1
) -> Transformer:
   # Embedding layers
   input_embedding_layer = Embeddings(d_model = d_model, vocab_size = input_vocab_size)
   output_embedding_layer = Embeddings(d_model = d_model, vocab_size = output_vocab_size)

   # Positional Encoding Layers
   input_pos_encoding_layer = PositionalEncoding(d_model = d_model, max_seq_len = input_seq_len, dropout = dropout)
   output_pos_encoding_layer = PositionalEncoding(d_model = d_model, max_seq_len = output_seq_len, dropout = dropout)

   # Encoder Layers
   encoder_layers = []
   for _ in range(num_encoding_layers):
      enc_attention = MultiHeadAttentionBlock(d_model = d_model, num_heads = num_attention_heads, dropout = dropout)
      enc_feed_forward = FeedForwardBlock(d_model = d_model, d_ff = d_ff, dropout = dropout)
      enc_layer = EncoderLayer(features = d_model, self_attention = enc_attention, feed_forward = enc_feed_forward, dropout = dropout)
      encoder_layers.append(enc_layer)
   
   # Decoder Layers
   decoder_layers = []
   for _ in range(num_decoding_layers):
      dec_self_attention = MultiHeadAttentionBlock(d_model = d_model, num_heads = num_attention_heads, dropout = dropout)
      dec_cross_attention = MultiHeadAttentionBlock(d_model = d_model, num_heads = num_attention_heads, dropout = dropout)
      dec_feed_forward = FeedForwardBlock(d_model = d_model, d_ff = d_ff, dropout = dropout)
      dec_layer = DecoderLayer(features = d_model, self_attention = dec_self_attention, cross_attention = dec_cross_attention, feed_forward = dec_feed_forward, dropout = dropout)
      decoder_layers.append(dec_layer)
   
   # Encoder - Decoder
   encoder = Encoder(encoder_layers)
   decoder = Decoder(decoder_layers)
   
   # Last layer :: Embedding -> Tokens Projection Layer
   embed_to_token_projection_layer = ProjectionLayer(d_model = d_model, vocab_size = output_vocab_size)

   model = Transformer(
      encoder = encoder,
      decoder = decoder,
      input_embedding = input_embedding_layer,
      output_embedding = output_embedding_layer,
      input_positional_encoding = input_pos_encoding_layer,
      output_positional_encoding = output_pos_encoding_layer,
      embeddings_to_tokens = embed_to_token_projection_layer
   )

   return model