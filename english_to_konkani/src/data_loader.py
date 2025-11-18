"""
Data loader for English-Konkani parallel corpus.
Reads from konkani_pairs.txt with eng:/kon: prefix format.
"""

import os
import re
from typing import List, Tuple, Optional
import numpy as np


class KonkaniPairsLoader:
    """
    Loads English-Konkani translation pairs from text file.
    Format: Lines alternating between 'eng:' and 'kon:' prefixes.
    """
    
    def __init__(self, pairs_file: str = "konkani_pairs.txt"):
        """
        Initialize the loader.
        
        Args:
            pairs_file: Path to the konkani_pairs.txt file
        """
        self.pairs_file = pairs_file
        self.eng_sentences = []
        self.kon_sentences = []
        
    def load_pairs(self) -> Tuple[List[str], List[str]]:
        """
        Load all translation pairs from file.
        
        Returns:
            Tuple of (english_sentences, konkani_sentences)
        """
        if not os.path.exists(self.pairs_file):
            raise FileNotFoundError(f"Pairs file not found: {self.pairs_file}")
        
        eng_sentences = []
        kon_sentences = []
        current_eng = None
        
        with open(self.pairs_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Parse eng: prefix
                if line.startswith('eng:'):
                    current_eng = line[4:].strip()
                
                # Parse kon: prefix
                elif line.startswith('kon:'):
                    current_kon = line[4:].strip()
                    
                    # Add pair only if we have both eng and kon
                    if current_eng is not None:
                        eng_sentences.append(current_eng)
                        kon_sentences.append(current_kon)
                        current_eng = None
        
        self.eng_sentences = eng_sentences
        self.kon_sentences = kon_sentences
        
        print(f"Loaded {len(eng_sentences)} translation pairs from {self.pairs_file}")
        return eng_sentences, kon_sentences
    
    def add_pair(self, english: str, konkani: str):
        """
        Add a new translation pair to the file (incremental learning).
        
        Args:
            english: English sentence
            konkani: Konkani translation
        """
        with open(self.pairs_file, 'a', encoding='utf-8') as f:
            # Add blank line if file is not empty
            if os.path.getsize(self.pairs_file) > 0:
                f.write('\n')
            
            f.write(f'eng:{english}\n')
            f.write(f'kon:{konkani}\n')
        
        # Update internal lists
        self.eng_sentences.append(english)
        self.kon_sentences.append(konkani)
        
        print(f"Added new pair: '{english}' -> '{konkani}'")
    
    def add_pairs_batch(self, pairs: List[Tuple[str, str]]):
        """
        Add multiple translation pairs at once.
        
        Args:
            pairs: List of (english, konkani) tuples
        """
        with open(self.pairs_file, 'a', encoding='utf-8') as f:
            for english, konkani in pairs:
                if os.path.getsize(self.pairs_file) > 0:
                    f.write('\n')
                
                f.write(f'eng:{english}\n')
                f.write(f'kon:{konkani}\n')
                
                self.eng_sentences.append(english)
                self.kon_sentences.append(konkani)
        
        print(f"Added {len(pairs)} new translation pairs")
    
    def get_statistics(self) -> dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.eng_sentences:
            self.load_pairs()
        
        eng_vocab = set()
        kon_vocab = set()
        
        for eng in self.eng_sentences:
            eng_vocab.update(eng.lower().split())
        
        for kon in self.kon_sentences:
            kon_vocab.update(kon.split())
        
        return {
            'num_pairs': len(self.eng_sentences),
            'eng_vocab_size': len(eng_vocab),
            'kon_vocab_size': len(kon_vocab),
            'avg_eng_length': np.mean([len(s.split()) for s in self.eng_sentences]),
            'avg_kon_length': np.mean([len(s.split()) for s in self.kon_sentences]),
        }


class FileBasedInputReader:
    """
    Reads input sentences from ISL model output file for translation.
    """
    
    def __init__(self, input_file: str = "isl_to_english.txt"):
        """
        Initialize the reader.
        
        Args:
            input_file: Path to file containing English sentences from ISL model
        """
        self.input_file = input_file
        self.last_position = 0
    
    def read_new_lines(self) -> List[str]:
        """
        Read new lines added to the input file since last read.
        
        Returns:
            List of new English sentences to translate
        """
        if not os.path.exists(self.input_file):
            return []
        
        new_sentences = []
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            # Seek to last position
            f.seek(self.last_position)
            
            # Read new lines
            for line in f:
                line = line.strip()
                if line:
                    new_sentences.append(line)
            
            # Update position
            self.last_position = f.tell()
        
        return new_sentences
    
    def read_all(self) -> List[str]:
        """
        Read all sentences from the input file.
        
        Returns:
            List of all English sentences
        """
        if not os.path.exists(self.input_file):
            return []
        
        sentences = []
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)
        
        return sentences
    
    def reset(self):
        """Reset the file position to beginning."""
        self.last_position = 0


if __name__ == "__main__":
    # Test the loader
    loader = KonkaniPairsLoader()
    
    try:
        eng, kon = loader.load_pairs()
        
        print(f"\nFirst 3 pairs:")
        for i in range(min(3, len(eng))):
            print(f"  {eng[i]} -> {kon[i]}")
        
        stats = loader.get_statistics()
        print(f"\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except FileNotFoundError:
        print("No konkani_pairs.txt found. Please create one first.")
