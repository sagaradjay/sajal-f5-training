import pandas as pd
from openai import OpenAI
import time
from typing import List, Dict
import re
from tqdm import tqdm
import random
import os
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class OpenAITransliterator:
    def __init__(self, model_name: str = "gpt-4o-mini", 
                 api_key: str = None,
                 max_tokens: int = 512,
                 temperature: float = 0.1,
                 max_workers: int = 10,
                 use_batching: bool = True):
        """
        Initialize OpenAI client for transliteration
        
        Args:
            model_name: OpenAI model name (gpt-4o-mini, gpt-4o, gpt-3.5-turbo, etc.)
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY environment variable)
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation (0.1 for consistency)
            max_workers: Number of concurrent workers for parallel processing
            use_batching: Whether to use OpenAI's batch API for cost savings
        """
        print(f"Initializing OpenAI client with model: {model_name}")
        
        # Initialize OpenAI client
        if api_key:
            self.client = OpenAI(api_key=api_key)
            self.api_key = api_key
        else:
            # Will use OPENAI_API_KEY environment variable
            self.client = OpenAI()
            self.api_key = os.getenv('OPENAI_API_KEY')
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_workers = max_workers
        self.use_batching = use_batching
        
        print("OpenAI client initialized successfully!")
    
    def create_multi_text_prompt(self, hindi_texts: List[str]) -> List[Dict]:
        """
        Create a prompt for multiple texts at once with better formatting for order preservation
        
        Args:
            hindi_texts: List of Hindi/Hinglish texts
            
        Returns:
            Messages list for OpenAI API
        """
        system_prompt = """You are an expert Hindi to English transliteration assistant. Your task is to convert Hindi (Devanagari script) words to English (Roman script) using SIMPLE transliteration without any diacritical marks.

Rules:
1. Only transliterate Hindi/Devanagari script to Roman letters
2. Keep English words exactly as they are
3. Maintain the original word order and structure
4. Use simple transliteration WITHOUT diacritical marks (no ā, ī, ū, ṅ, ṃ, etc.)
5. Use standard English letters only (a, e, i, o, u, etc.)
6. Output ONLY the transliterated texts, each on a new line, in the EXACT same order as input
7. Do not add any explanations, numbers, or extra text

IMPORTANT: You will receive numbered inputs. Return the transliterations in the same order, one per line, without numbers.

Examples:
Input:
1. दोस्तों bash में nested और multilevel
2. यह tutorial में हम निम्न के बारे में सीखेंगे
3. गीतकार पिन्टू गिरी के लिखे गीत को संगीत से संवारा है

Output:
dosto bash mein nested aur multilevel
yah tutorial mein hum nimn ke baare mein seekhenge
geetkaar pintu giri ke likhe geet ko sangeet se savaraa hai
"""
        
        # Create numbered input for better order tracking
        numbered_texts = []
        for i, text in enumerate(hindi_texts, 1):
            numbered_texts.append(f"{i}. {text}")
        
        user_prompt = f"Transliterate these numbered texts (return in same order, one per line, no numbers):\n\n" + "\n".join(numbered_texts)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return messages
    def create_transliteration_prompt(self, hindi_text: str) -> List[Dict]:
        """
        Create a proper prompt for OpenAI model with simple transliteration (single text)
        
        Args:
            hindi_text: Input Hindi/Hinglish text
            
        Returns:
            Messages list for OpenAI API
        """
        system_prompt = """You are an expert Hindi to English transliteration assistant. Your task is to convert Hindi (Devanagari script) words to English (Roman script) using SIMPLE transliteration without any diacritical marks.

Rules:
1. Only transliterate Hindi/Devanagari script to Roman letters
2. Keep English words exactly as they are
3. Maintain the original word order and structure
4. Use simple transliteration WITHOUT diacritical marks (no ā, ī, ū, ṅ, ṃ, etc.)
5. Use standard English letters only (a, e, i, o, u, etc.)
6. Output only the transliterated text, nothing else

Examples:
- Input: "दोस्तों bash में nested और multilevel"
- Output: "dosto bash mein nested aur multilevel"

- Input: "यह tutorial में हम निम्न के बारे में सीखेंगे" 
- Output: "yah tutorial mein hum nimn ke baare mein seekhenge"

- Input: "गीतकार पिन्टू गिरी के लिखे गीत को संगीत से संवारा है"
- Output: "geetkaar pintu giri ke likhe geet ko sangeet se savaraa hai"
"""
        
        user_prompt = f"Transliterate this text: {hindi_text}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return messages
        
    def transliterate_multiple_texts(self, texts: List[str]) -> List[str]:
        """
        Transliterate multiple texts in a single API call with better error handling
        
        Args:
            texts: List of Hindi/Hinglish texts (recommended 3-5 texts per call)
            
        Returns:
            List of transliterated texts in same order
        """
        if not texts:
            return []
            
        messages = self.create_multi_text_prompt(texts)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens * len(texts),  # Scale tokens based on number of texts
                temperature=self.temperature,
                top_p=0.9
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            # Split the response by lines and clean each
            lines = generated_text.split('\n')
            results = []
            
            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    # Remove any numbering that might have been added
                    line = re.sub(r'^\d+\.\s*', '', line)
                    line = re.sub(r'^\d+\)\s*', '', line)
                    line = re.sub(r'^\d+\s*[-:]\s*', '', line)
                    cleaned = self.clean_output(line)
                    if cleaned:
                        results.append(cleaned)
            
            # Ensure we have exactly the same number of results as inputs
            if len(results) != len(texts):
                print(f"Warning: Expected {len(texts)} results, got {len(results)}. Using fallback.")
                # If count mismatch, fall back to single-text processing
                return [self.transliterate_single(text) for text in texts]
            
            return results
            
        except Exception as e:
            print(f"Error in multi-text API call: {e}")
            # Fallback to single-text processing if multi-text fails
            try:
                return [self.transliterate_single(text) for text in texts]
            except:
                # Last resort: return original texts
                return texts
    
    def transliterate_single(self, text: str) -> str:
        """
        Transliterate a single text using OpenAI API
        
        Args:
            text: Hindi/Hinglish text
            
        Returns:
            Transliterated text
        """
        messages = self.create_transliteration_prompt(text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9
            )
            
            generated_text = response.choices[0].message.content.strip()
            cleaned_text = self.clean_output(generated_text)
            return cleaned_text
            
        except Exception as e:
            print(f"Error in API call: {e}")
    def transliterate_batch(self, texts: List[str]) -> List[str]:
        """
        Transliterate a batch of texts - chooses between parallel or sequential based on batch size
        
        Args:
            texts: List of Hindi/Hinglish texts
            
        Returns:
            List of transliterated texts
        """
        if len(texts) > 5:
            # Use parallel processing for larger batches
            return self.transliterate_batch_parallel(texts)
        else:
            # Use sequential for small batches to avoid overhead
            return self.transliterate_multiple_texts(texts)
    
    def transliterate_batch_parallel(self, texts: List[str]) -> List[str]:
        """
        Transliterate a batch of texts using parallel processing with order preservation
        
        Args:
            texts: List of Hindi/Hinglish texts
            
        Returns:
            List of transliterated texts in the same order as input
        """
        if not texts:
            return []
        
        # Group texts into chunks for multi-text API calls
        chunk_size = 5  # Smaller chunks for better order control
        chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # Create ordered results list
        results = [None] * len(texts)
        
        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as executor:
            # Submit all chunks with their indices for order preservation
            future_to_index = {}
            for chunk_idx, chunk in enumerate(chunks):
                future = executor.submit(self.transliterate_multiple_texts, chunk)
                future_to_index[future] = chunk_idx
            
            # Collect results and maintain order
            for future in as_completed(future_to_index):
                chunk_idx = future_to_index[future]
                start_idx = chunk_idx * chunk_size
                
                try:
                    chunk_results = future.result()
                    # Place results in correct positions
                    for i, result in enumerate(chunk_results):
                        if start_idx + i < len(results):
                            results[start_idx + i] = result
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx}: {e}")
                    # Use original texts if chunk fails
                    chunk = chunks[chunk_idx]
                    for i, original_text in enumerate(chunk):
                        if start_idx + i < len(results):
                            results[start_idx + i] = original_text
        
        # Fill any None values with original text (safety check)
        for i, result in enumerate(results):
            if result is None:
                results[i] = texts[i]
        
        return results
    
    def clean_output(self, text: str) -> str:
        """
        Clean the model output to extract only the transliterated text
        
        Args:
            text: Raw model output
            
        Returns:
            Cleaned transliterated text
        """
        # Remove any potential system tokens or extra text
        text = text.strip()
        
        # If the model adds extra explanation, extract just the transliteration
        lines = text.split('\n')
        if lines:
            # Usually the first line contains the transliteration
            cleaned = lines[0].strip()
            
            # Remove common prefixes the model might add
            prefixes_to_remove = [
                "Output:", "Transliterated:", "Result:", 
                "Translation:", "Transliteration:", "Text:"
            ]
            
            for prefix in prefixes_to_remove:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            
            return cleaned
        
        return text
    
    def cleanup(self):
        """
        Clean up resources (no GPU cleanup needed for OpenAI API)
        """
        print("Cleanup completed (no resources to clean for OpenAI API)")
    
    def has_hindi_text(self, text: str) -> bool:
        """
        Check if text contains Hindi/Devanagari script
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains Hindi characters
        """
        # Hindi/Devanagari Unicode range: U+0900 to U+097F
        hindi_pattern = re.compile(r'[\u0900-\u097F]')
        return bool(hindi_pattern.search(text))
    
    def is_single_word(self, text: str) -> bool:
        """
        Check if text is a single word (no spaces)
        
        Args:
            text: Text to check
            
        Returns:
            True if text is a single word
        """
        return len(text.strip().split()) <= 1
    
    def print_batch_examples(self, batch_texts: List[str], batch_results: List[str], batch_num: int):
        """
        Print examples from the current batch
        
        Args:
            batch_texts: Original texts from the batch
            batch_results: Transliterated texts from the batch
            batch_num: Current batch number
        """
        print(f"\n=== Batch {batch_num} Examples ===")
        
        # Show up to 3 examples from this batch
        num_examples = min(3, len(batch_texts))
        
        for i in range(num_examples):
            print(f"\nExample {i+1}:")
            print(f"  Original: {batch_texts[i]}")
            print(f"  Transliterated: {batch_results[i]}")
            print("-" * 40)
        
        print(f"Batch {batch_num} completed: {len(batch_texts)} texts processed")
    
    def process_metadata_csv(self, input_file: str, 
                           output_file: str = None,
                           num_rows_to_transliterate: int = 20000,
                           batch_size: int = 10,  # Smaller batch size for API rate limits
                           text_column: str = "text") -> str:
        """
        Process the metadata CSV file and transliterate random rows
        
        Args:
            input_file: Input CSV file path
            output_file: Output CSV file path (if None, auto-generated)
            num_rows_to_transliterate: Number of random rows to transliterate
            batch_size: Batch size for processing (smaller for API rate limits)
            text_column: Column containing text to transliterate
            
        Returns:
            Path to the output file
        """
        print(f"Loading CSV file: {input_file}")
        
        # Read CSV with pipe separator
        df = pd.read_csv(input_file, sep='|')
        print(f"Loaded {len(df)} rows from CSV")
        print(f"Columns: {df.columns.tolist()}")
        
        # Filter rows that have Hindi text and are not single words
        hindi_mask = df[text_column].apply(self.has_hindi_text)
        multiword_mask = df[text_column].apply(lambda x: not self.is_single_word(str(x)))
        
        # Combine masks
        valid_mask = hindi_mask & multiword_mask
        
        valid_indices = df[valid_mask].index.tolist()
        print(f"Found {len(valid_indices)} rows with multi-word Hindi text")
        
        if len(valid_indices) < num_rows_to_transliterate:
            print(f"Warning: Only {len(valid_indices)} valid rows found, but {num_rows_to_transliterate} requested")
            num_rows_to_transliterate = len(valid_indices)
        
        # Randomly select rows to transliterate
        random.seed(42)  # For reproducibility
        selected_indices = random.sample(valid_indices, num_rows_to_transliterate)
        print(f"Selected {len(selected_indices)} random rows for transliteration")
        
        # Create a copy of the dataframe
        df_output = df.copy()
        
        # Process selected rows in batches
        texts_to_transliterate = df.loc[selected_indices, text_column].tolist()
        
        print(f"Processing {len(texts_to_transliterate)} texts in batches of {batch_size}...")
        
        all_transliterated = []
        batch_num = 1
        
        # Process in batches
        for i in tqdm(range(0, len(texts_to_transliterate), batch_size), desc="Processing batches"):
            batch_texts = texts_to_transliterate[i:i+batch_size]
            transliterated_batch = self.transliterate_batch(batch_texts)
            all_transliterated.extend(transliterated_batch)
            
            # Print examples after each batch
            self.print_batch_examples(batch_texts, transliterated_batch, batch_num)
            batch_num += 1
            
            # No need for delays with parallel processing and multi-text API calls
        
        # Update the dataframe with transliterated text
        for idx, transliterated_text in zip(selected_indices, all_transliterated):
            df_output.loc[idx, text_column] = transliterated_text
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = input_file.replace('.csv', '')
            output_file = f"{base_name}_transliterated.csv"
        
        # Save the updated dataframe
        df_output.to_csv(output_file, sep='|', index=False)
        print(f"\nTransliterated CSV saved as: {output_file}")
        
        # Show final summary
        print(f"\n=== Final Summary ===")
        print(f"Total rows processed: {len(all_transliterated)}")
        print(f"Total batches: {batch_num - 1}")
        print(f"Output file: {output_file}")
        
        return output_file

def validate_paths(input_file: str, output_file: str) -> bool:
    """
    Validate input and output file paths before processing
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        
    Returns:
        True if paths are valid, False otherwise
    """
    print("=== Validating File Paths ===")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ ERROR: Input file does not exist: {input_file}")
        return False
    
    # Check if input file is readable
    if not os.access(input_file, os.R_OK):
        print(f"❌ ERROR: Input file is not readable: {input_file}")
        return False
    
    # Check if input file is a CSV file
    if not input_file.lower().endswith('.csv'):
        print(f"❌ ERROR: Input file is not a CSV file: {input_file}")
        return False
    
    print(f"✅ Input file is valid: {input_file}")
    
    # Check if output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        print(f"❌ ERROR: Output directory does not exist: {output_dir}")
        return False
    
    # Check if output directory is writable
    if not os.access(output_dir, os.W_OK):
        print(f"❌ ERROR: Output directory is not writable: {output_dir}")
        return False
    
    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"⚠️  WARNING: Output file already exists: {output_file}")
        response = input("Do you want to overwrite it? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("❌ Operation cancelled by user")
            return False
    
    print(f"✅ Output path is valid: {output_file}")
    
    # Check file sizes
    try:
        input_size = os.path.getsize(input_file)
        print(f"📊 Input file size: {input_size:,} bytes ({input_size / (1024*1024):.2f} MB)")
        
        if input_size == 0:
            print(f"❌ ERROR: Input file is empty: {input_file}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: Could not get input file size: {e}")
        return False
    
    print("✅ All path validations passed!")
    return True

def main():
    # Configuration
    MODEL_NAME = "gpt-4o-mini"  # You can use "gpt-4o", "gpt-3.5-turbo", etc.
    BATCH_SIZE = 50  # Reduced batch size for better order control
    NUM_ROWS_TO_TRANSLITERATE = 20000
    MAX_WORKERS = 10  # Reduced workers for better order control
    
    # OpenAI API Key - set this or use environment variable OPENAI_API_KEY
    API_KEY = ""
    
    # File paths
    INPUT_FILE = "/media/rdp/New Volume/F5-TTS/Combined_Hindi_TTS_Raw_Data/metadata.csv"
    OUTPUT_FILE = "/media/rdp/New Volume/F5-TTS/Combined_Hindi_TTS_Raw_Data/metadata_transliterated.csv"
    
    # Validate paths before proceeding
    if not validate_paths(INPUT_FILE, OUTPUT_FILE):
        print("❌ Path validation failed. Exiting...")
        return
    
    # Initialize transliterator
    try:
        transliterator = OpenAITransliterator(
            model_name=MODEL_NAME,
            api_key=API_KEY,  # If None, will use OPENAI_API_KEY environment variable
            temperature=0.1,  # Low temperature for consistency
            max_tokens=512,
            max_workers=MAX_WORKERS
        )
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print("Make sure you have openai installed: pip install openai")
        print("And set your API key either as parameter or OPENAI_API_KEY environment variable")
        return
    
    # Test with a single example first
    test_sentences = [
        "इन तारों को लीड्स कहते हैं",
        "परीक्षा में स्लीवलेस नथनी ईयरिंग माला और साइड के बटन तक निकलवा लिए गए",
        "चश्मदीदों ने बताया कि इस हादसे के बाद पटरियों के दोनों ओर खून से सनी लाशें पड़ीं थीं",
        "गीतकार पिन्टू गिरी के लिखे गीत को संगीत से संवारा है संगीतकार सावन कुमार ने"
    ]
    
    print("\n=== Testing transliteration ===")
    results = transliterator.transliterate_batch(test_sentences)
    
    for original, transliterated in zip(test_sentences, results):
        print(f"Original: {original}")
        print(f"Transliterated: {transliterated}")
        print("-" * 50)
    
    # Process the metadata CSV
    print(f"\n=== Processing metadata CSV ===")
    try:
        output_file = transliterator.process_metadata_csv(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            num_rows_to_transliterate=NUM_ROWS_TO_TRANSLITERATE,
            batch_size=BATCH_SIZE
        )
        
        print(f"\nSuccessfully processed CSV!")
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing CSV: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    try:
        transliterator.cleanup()
    except Exception as e:
        print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()