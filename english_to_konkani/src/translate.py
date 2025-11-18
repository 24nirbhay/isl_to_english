"""
Translation inference module with file-based I/O.
Reads English sentences from isl_to_english.txt and outputs Konkani translations.
"""

import os
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np

from tokenizer import SimpleTokenizer


class FileBasedTranslator:
    """
    Translates English to Konkani using trained model.
    Monitors input file and writes translations to output.
    """
    
    def __init__(
        self,
        model_path: str = "models/master/model.keras",
        tokenizer_eng_path: str = "models/master/tokenizer_eng.pkl",
        tokenizer_kon_path: str = "models/master/tokenizer_kon.pkl",
        input_file: str = "isl_to_english.txt",
        output_file: str = "english_to_konkani.txt",
        max_eng_length: int = 100,
        max_kon_length: int = 100
    ):
        """
        Initialize translator.
        
        Args:
            model_path: Path to trained model
            tokenizer_eng_path: Path to English tokenizer
            tokenizer_kon_path: Path to Konkani tokenizer
            input_file: Input file with English sentences
            output_file: Output file for Konkani translations
            max_eng_length: Maximum English sequence length
            max_kon_length: Maximum Konkani sequence length
        """
        self.input_file = input_file
        self.output_file = output_file
        self.max_eng_length = max_eng_length
        self.max_kon_length = max_kon_length
        
        print("Loading translation model...")
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = keras.models.load_model(model_path, compile=False)
        print(f"Model loaded from {model_path}")
        
        # Load tokenizers
        self.eng_tokenizer = SimpleTokenizer()
        self.eng_tokenizer.load(tokenizer_eng_path)
        
        self.kon_tokenizer = SimpleTokenizer()
        self.kon_tokenizer.load(tokenizer_kon_path)
        
        print("Tokenizers loaded")
        print(f"English vocab: {self.eng_tokenizer.vocab_size}")
        print(f"Konkani vocab: {self.kon_tokenizer.vocab_size}")
        
        self.last_file_position = 0
    
    def translate(self, english_text: str) -> str:
        """
        Translate a single English sentence to Konkani.
        
        Args:
            english_text: English sentence
            
        Returns:
            Konkani translation
        """
        # Encode English input
        encoder_input = np.array([
            self.eng_tokenizer.encode(
                english_text,
                max_length=self.max_eng_length,
                add_special=False
            )
        ])
        
        # Start with START token
        decoder_input = [self.kon_tokenizer.char_to_idx[self.kon_tokenizer.START_TOKEN]]
        
        # Generate translation token by token
        for _ in range(self.max_kon_length - 1):
            # Pad decoder input to max length
            padded_decoder_input = decoder_input + [
                self.kon_tokenizer.char_to_idx[self.kon_tokenizer.PAD_TOKEN]
            ] * (self.max_kon_length - 1 - len(decoder_input))
            
            decoder_input_array = np.array([padded_decoder_input])
            
            # Predict next token
            predictions = self.model.predict([encoder_input, decoder_input_array], verbose=0)
            
            # Get the last predicted token
            next_token_probs = predictions[0, len(decoder_input) - 1, :]
            next_token = np.argmax(next_token_probs)
            
            # Stop if END token is predicted
            if next_token == self.kon_tokenizer.char_to_idx[self.kon_tokenizer.END_TOKEN]:
                break
            
            decoder_input.append(next_token)
        
        # Decode to text
        translation = self.kon_tokenizer.decode(decoder_input, skip_special=True)
        
        return translation
    
    def translate_batch(self, english_sentences: list) -> list:
        """
        Translate multiple sentences.
        
        Args:
            english_sentences: List of English sentences
            
        Returns:
            List of Konkani translations
        """
        translations = []
        
        for sentence in english_sentences:
            translation = self.translate(sentence)
            translations.append(translation)
        
        return translations
    
    def monitor_file(self, check_interval: float = 1.0):
        """
        Monitor input file for new sentences and translate them.
        
        Args:
            check_interval: Time in seconds between file checks
        """
        print(f"\nMonitoring {self.input_file} for new sentences...")
        print(f"Translations will be written to {self.output_file}")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Check if input file exists
                if not os.path.exists(self.input_file):
                    time.sleep(check_interval)
                    continue
                
                # Read new lines
                new_sentences = self._read_new_lines()
                
                if new_sentences:
                    print(f"Found {len(new_sentences)} new sentence(s)")
                    
                    # Translate and write
                    for sentence in new_sentences:
                        translation = self.translate(sentence)
                        self._write_translation(sentence, translation)
                        print(f"  {sentence} -> {translation}")
                
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            print("\nStopped monitoring")
    
    def _read_new_lines(self) -> list:
        """Read new lines from input file."""
        new_sentences = []
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            # Seek to last position
            f.seek(self.last_file_position)
            
            # Read new lines
            for line in f:
                line = line.strip()
                if line:
                    new_sentences.append(line)
            
            # Update position
            self.last_file_position = f.tell()
        
        return new_sentences
    
    def _write_translation(self, english: str, konkani: str):
        """Write translation to output file."""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(f"{english} | {konkani}\n")
    
    def translate_file(self, input_path: str = None, output_path: str = None):
        """
        Translate all sentences in a file at once.
        
        Args:
            input_path: Input file path (defaults to self.input_file)
            output_path: Output file path (defaults to self.output_file)
        """
        input_path = input_path or self.input_file
        output_path = output_path or self.output_file
        
        if not os.path.exists(input_path):
            print(f"Input file not found: {input_path}")
            return
        
        # Read all sentences
        sentences = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)
        
        if not sentences:
            print("No sentences found in input file")
            return
        
        print(f"Translating {len(sentences)} sentences...")
        
        # Translate
        translations = self.translate_batch(sentences)
        
        # Write to output
        with open(output_path, 'w', encoding='utf-8') as f:
            for eng, kon in zip(sentences, translations):
                f.write(f"{eng} | {kon}\n")
        
        print(f"Translations written to {output_path}")
        
        # Show sample
        print("\nSample translations:")
        for i, (eng, kon) in enumerate(zip(sentences[:5], translations[:5])):
            print(f"  {eng} -> {kon}")


if __name__ == "__main__":
    # Test translator
    translator = FileBasedTranslator(
        model_path="../models/master/model.keras",
        tokenizer_eng_path="../models/master/tokenizer_eng.pkl",
        tokenizer_kon_path="../models/master/tokenizer_kon.pkl",
        input_file="../isl_to_english.txt",
        output_file="../english_to_konkani.txt"
    )
    
    # Test single translation
    test_sentence = "hello"
    translation = translator.translate(test_sentence)
    print(f"\nTest: {test_sentence} -> {translation}")
    
    # Monitor file (uncomment to use)
    # translator.monitor_file()
