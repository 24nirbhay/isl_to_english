"""
Simplified Transformer-based Neural Machine Translation model.
English to Konkani translation - beginner-friendly architecture.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class PositionalEncoding(layers.Layer):
    """
    Adds positional information to input embeddings.
    Helps the model understand word order.
    """
    
    def __init__(self, max_length, embed_dim):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        
        # Create positional encoding matrix
        position = np.arange(max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        
        pos_encoding = np.zeros((max_length, embed_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        self.pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
    
    def call(self, x):
        """Add positional encoding to embeddings."""
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:seq_len, :]


class TransformerEncoder(layers.Layer):
    """
    Encoder block with multi-head attention and feed-forward network.
    Processes English input sequences.
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(self, x, training=False):
        """Forward pass through encoder block."""
        # Multi-head attention
        attn_output = self.attention(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)
        
        return x


class TransformerDecoder(layers.Layer):
    """
    Decoder block with masked self-attention and cross-attention.
    Generates Konkani output sequences.
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.cross_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.layernorm3 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)
    
    def call(self, x, encoder_output, training=False, look_ahead_mask=None):
        """Forward pass through decoder block."""
        # Masked self-attention
        attn1 = self.self_attention(
            query=x, value=x, key=x,
            attention_mask=look_ahead_mask,
            training=training
        )
        attn1 = self.dropout1(attn1, training=training)
        x = self.layernorm1(x + attn1)
        
        # Cross-attention with encoder output
        attn2 = self.cross_attention(
            query=x, value=encoder_output, key=encoder_output,
            training=training
        )
        attn2 = self.dropout2(attn2, training=training)
        x = self.layernorm2(x + attn2)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        ffn_output = self.dropout3(ffn_output, training=training)
        x = self.layernorm3(x + ffn_output)
        
        return x


def create_look_ahead_mask(size):
    """
    Create mask to prevent attending to future tokens.
    Used in decoder self-attention.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_transformer_model(
    eng_vocab_size: int,
    kon_vocab_size: int,
    max_eng_length: int = 100,
    max_kon_length: int = 100,
    embed_dim: int = 128,
    num_heads: int = 4,
    ff_dim: int = 512,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    dropout: float = 0.1
):
    """
    Create a Transformer model for English-to-Konkani translation.
    
    Simplified architecture for beginners:
    - Character-level tokenization
    - Smaller embedding dimension (128 vs 512 in original)
    - Fewer layers (2 vs 6 in original)
    - Easy to train on CPU/small GPU
    
    Args:
        eng_vocab_size: English vocabulary size
        kon_vocab_size: Konkani vocabulary size
        max_eng_length: Maximum English sequence length
        max_kon_length: Maximum Konkani sequence length
        embed_dim: Embedding dimension (default: 128)
        num_heads: Number of attention heads (default: 4)
        ff_dim: Feed-forward dimension (default: 512)
        num_encoder_layers: Number of encoder layers (default: 2)
        num_decoder_layers: Number of decoder layers (default: 2)
        dropout: Dropout rate (default: 0.1)
    
    Returns:
        Keras Model
    """
    
    # Encoder input
    encoder_input = keras.Input(shape=(max_eng_length,), name='encoder_input')
    
    # Decoder input
    decoder_input = keras.Input(shape=(max_kon_length - 1,), name='decoder_input')
    
    # Encoder embedding
    encoder_embedding = layers.Embedding(
        input_dim=eng_vocab_size,
        output_dim=embed_dim,
        mask_zero=True
    )(encoder_input)
    encoder_embedding = PositionalEncoding(max_eng_length, embed_dim)(encoder_embedding)
    
    # Encoder layers
    encoder_output = encoder_embedding
    for _ in range(num_encoder_layers):
        encoder_output = TransformerEncoder(embed_dim, num_heads, ff_dim, dropout)(encoder_output)
    
    # Decoder embedding
    decoder_embedding = layers.Embedding(
        input_dim=kon_vocab_size,
        output_dim=embed_dim,
        mask_zero=True
    )(decoder_input)
    decoder_embedding = PositionalEncoding(max_kon_length, embed_dim)(decoder_embedding)
    
    # Create look-ahead mask for decoder
    look_ahead_mask = create_look_ahead_mask(max_kon_length - 1)
    
    # Decoder layers
    decoder_output = decoder_embedding
    for _ in range(num_decoder_layers):
        decoder_output = TransformerDecoder(embed_dim, num_heads, ff_dim, dropout)(
            decoder_output, encoder_output, look_ahead_mask=look_ahead_mask
        )
    
    # Final output layer
    output = layers.Dense(kon_vocab_size, activation='softmax', name='output')(decoder_output)
    
    # Create model
    model = keras.Model(
        inputs=[encoder_input, decoder_input],
        outputs=output,
        name='transformer_nmt'
    )
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_transformer_model(
        eng_vocab_size=100,
        kon_vocab_size=150,
        max_eng_length=50,
        max_kon_length=50,
        embed_dim=64,  # Smaller for testing
        num_heads=2,
        ff_dim=256,
        num_encoder_layers=1,
        num_decoder_layers=1
    )
    
    model.summary()
    print("\nModel created successfully!")
    print(f"Total parameters: {model.count_params():,}")
