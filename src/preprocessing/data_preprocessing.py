import re
import pandas as pd
from typing import Dict, List
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from pathlib import Path
import logging
from tqdm import tqdm
import gc
import time
import psutil
import os

# Setup logging with file handler
log_file = 'preprocessing.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self, use_stemming: bool = True):
        """Initialize the text preprocessor."""
        self.use_stemming = use_stemming
        self.slang_dict = self._load_slang_dict()
        
        if use_stemming:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
    
    def _load_slang_dict(self) -> Dict[str, str]:
        """Load slang dictionary from CSV file."""
        slang_df = pd.read_csv('data/raw/slang_words (2).csv')
        return dict(zip(slang_df['slang'], slang_df['meaning']))
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing URLs, emails, punctuation, and extra whitespace."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation except apostrophes within words
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _normalize_slang(self, text: str) -> str:
        """Replace slang words with their formal equivalents."""
        words = text.split()
        normalized_words = [self.slang_dict.get(word, word) for word in words]
        return ' '.join(normalized_words)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess a single text string."""
        # Clean text
        text = self._clean_text(text)
        
        # Normalize slang
        text = self._normalize_slang(text)
        
        # Apply stemming if enabled
        if self.use_stemming:
            text = self.stemmer.stem(text)
        
        return text

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def save_checkpoint(df: pd.DataFrame, batch_num: int, mode: str):
    """Save checkpoint file."""
    checkpoint_file = f"checkpoint_{mode}_batch_{batch_num}.csv"
    df.to_csv(checkpoint_file, index=False)
    logger.info(f"Saved checkpoint: {checkpoint_file}")
    logger.info(f"Memory usage: {get_memory_usage():.2f} MB")

def process_without_stemming(input_file: str = 'data/raw/clean_sarcasm_data.csv',
                           output_file: str = 'data/processed/preprocessed_sarcasm_data_no_stemming.csv'):
    """Process dataset without stemming."""
    BATCH_SIZE = 1000
    CHECKPOINT_INTERVAL = 10
    
    logger.info("Starting preprocessing without stemming...")
    start_time = time.time()
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(use_stemming=False)
    
    # Load data
    df = pd.read_csv(input_file)
    total_rows = len(df)
    total_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
    
    logger.info(f"Total rows: {total_rows}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Total batches: {total_batches}")
    
    # Initialize list for processed batches
    processed_batches = []
    
    # Process batches
    for i in tqdm(range(total_batches), desc="Processing without stemming"):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, total_rows)
        batch = df.iloc[start_idx:end_idx].copy()
        
        # Process batch
        batch['text'] = batch['text'].apply(preprocessor.preprocess_text)
        processed_batches.append(batch)
        
        # Save checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_df = pd.concat(processed_batches, ignore_index=True)
            save_checkpoint(checkpoint_df, i + 1, 'no_stemming')
            
        # Clear memory
        gc.collect()
    
    # Combine all batches and save final result
    final_df = pd.concat(processed_batches, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    
    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"Processing without stemming completed in {processing_time:.2f} seconds")
    logger.info(f"Final output saved to: {output_file}")

def process_with_stemming(input_file: str = 'data/raw/clean_sarcasm_data.csv',
                         output_file: str = 'data/processed/preprocessed_sarcasm_data_with_stemming.csv'):
    """Process dataset with stemming."""
    BATCH_SIZE = 250
    CHECKPOINT_INTERVAL = 5
    
    logger.info("Starting preprocessing with stemming...")
    start_time = time.time()
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(use_stemming=True)
    
    # Load data
    df = pd.read_csv(input_file)
    total_rows = len(df)
    total_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
    
    logger.info(f"Total rows: {total_rows}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Total batches: {total_batches}")
    
    # Initialize list for processed batches
    processed_batches = []
    
    # Process batches
    for i in tqdm(range(total_batches), desc="Processing with stemming"):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, total_rows)
        batch = df.iloc[start_idx:end_idx].copy()
        
        # Process batch
        batch['text'] = batch['text'].apply(preprocessor.preprocess_text)
        processed_batches.append(batch)
        
        # Save checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_df = pd.concat(processed_batches, ignore_index=True)
            save_checkpoint(checkpoint_df, i + 1, 'with_stemming')
            
        # Clear memory
        gc.collect()
    
    # Combine all batches and save final result
    final_df = pd.concat(processed_batches, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    
    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"Processing with stemming completed in {processing_time:.2f} seconds")
    logger.info(f"Final output saved to: {output_file}")

def main():
    # Print initial information
    logger.info("Starting preprocessing pipeline...")
    logger.info("Estimated processing times based on test run:")
    logger.info("- Without stemming: ~1 minute")
    logger.info("- With stemming: ~32 minutes")
    
    try:
        # Process without stemming first (faster)
        logger.info("Starting non-stemming preprocessing...")
        process_without_stemming()
        
        # Process with stemming
        logger.info("Starting stemming preprocessing...")
        process_with_stemming()
        
        logger.info("All preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 