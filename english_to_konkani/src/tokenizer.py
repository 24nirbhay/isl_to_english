"""
Text preprocessing and tokenization for English-Konkani NMT.
Simple character-level tokenization for beginner-friendliness.
"""

import pickle
import os
from typing import List, Tuple, Dict
import numpy as np


class SimpleTokenizer:
    """
    Simple character-level tokenizer for NMT.
    Beginner-friendly approach - treats each character as a token.
    """
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        self.UNK_TOKEN = '<UNK>'
        
    def fit(self, texts: List[str]):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of sentences
        """
        # Get all unique characters
        chars = set()
        for text in texts:
            chars.update(text.lower())
        
        # Add special tokens
        vocab = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]
        vocab.extend(sorted(list(chars)))
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(vocab)
        
        print(f"Vocabulary size: {self.vocab_size}")
        
    def encode(self, text: str, max_length: int = None, add_special: bool = True) -> List[int]:
        """
        Convert text to sequence of indices.
        
        Args:
            text: Input text
            max_length: Maximum sequence length (will pad/truncate)
            add_special: Whether to add START/END tokens
            
        Returns:
            List of character indices
        """
        text = text.lower()
        
        # Convert to indices
        indices = []
        
        if add_special:
            indices.append(self.char_to_idx[self.START_TOKEN])
        
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.char_to_idx[self.UNK_TOKEN])
        
        if add_special:
            indices.append(self.char_to_idx[self.END_TOKEN])
        
        # Pad or truncate
        if max_length:
            if len(indices) < max_length:
                indices.extend([self.char_to_idx[self.PAD_TOKEN]] * (max_length - len(indices)))
            else:
                indices = indices[:max_length]
        
        return indices
    
    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """
        Convert sequence of indices back to text.
        
        Args:
            indices: List of character indices
            skip_special: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        chars = []
        
        for idx in indices:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                
                # Skip special tokens if requested
                if skip_special and char in [self.PAD_TOKEN, self.START_TOKEN, 
                                              self.END_TOKEN, self.UNK_TOKEN]:
                    continue
                
                chars.append(char)
        
        return ''.join(chars)
    
    def save(self, filepath: str):
        """Save tokenizer to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'vocab_size': self.vocab_size
            }, f)
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load tokenizer from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.char_to_idx = data['char_to_idx']
            self.idx_to_char = data['idx_to_char']
            self.vocab_size = data['vocab_size']
        print(f"Tokenizer loaded from {filepath}")


def prepare_translation_data(
    eng_sentences: List[str],
    kon_sentences: List[str],
    max_eng_length: int = 100,
    max_kon_length: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SimpleTokenizer, SimpleTokenizer]:
    """
    Prepare data for training.
    
    Args:
        eng_sentences: English sentences
        kon_sentences: Konkani sentences
        max_eng_length: Maximum English sequence length
        max_kon_length: Maximum Konkani sequence length
        
    Returns:
        encoder_input, decoder_input, decoder_output, eng_tokenizer, kon_tokenizer
    """
    # Create tokenizers
    eng_tokenizer = SimpleTokenizer()
    kon_tokenizer = SimpleTokenizer()
    
    # Fit on data
    eng_tokenizer.fit(eng_sentences)
    kon_tokenizer.fit(kon_sentences)
    
    # Encode sequences
    encoder_input = []
    decoder_input = []
    decoder_output = []
    
    for eng, kon in zip(eng_sentences, kon_sentences):
        # Encoder input (English without special tokens)
        eng_seq = eng_tokenizer.encode(eng, max_length=max_eng_length, add_special=False)
        encoder_input.append(eng_seq)
        
        # Decoder input (Konkani with START token)
        kon_seq = kon_tokenizer.encode(kon, max_length=max_kon_length, add_special=True)
        decoder_input.append(kon_seq[:-1])  # Remove END token
        
        # Decoder output (Konkani shifted by 1, with END token)
        decoder_output.append(kon_seq[1:])  # Remove START token
    
    # Convert to numpy arrays
    encoder_input = np.array(encoder_input)
    decoder_input = np.array(decoder_input)
    decoder_output = np.array(decoder_output)
    
    print(f"Prepared {len(eng_sentences)} samples")
    print(f"Encoder input shape: {encoder_input.shape}")
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Decoder output shape: {decoder_output.shape}")
    
    return encoder_input, decoder_input, decoder_output, eng_tokenizer, kon_tokenizer


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = SimpleTokenizer()
    
    texts = ["hello", "नमस्कार", "good morning", "सुप्रभात"]
    tokenizer.fit(texts)
    
    # Test encode/decode
    for text in texts[:2]:
        encoded = tokenizer.encode(text, max_length=20)
        decoded = tokenizer.decode(encoded)
        print(f"{text} -> {encoded[:10]}... -> {decoded}")
