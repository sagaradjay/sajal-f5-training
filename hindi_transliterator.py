import pandas as pd
from vllm import LLM, SamplingParams
import time
from typing import List, Dict
import torch
import re
from tqdm import tqdm
import random
import numpy as np

class VLLMQwenTransliterator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
                 tensor_parallel_size: int = 1, 
                 gpu_memory_utilization: float = 0.8,
                 max_model_len: int = 4096):
        """
        Initialize vLLM with Qwen model for transliteration
        
        Args:
            model_name: Qwen model name from HuggingFace
            tensor_parallel_size: Number of GPUs to use
            gpu_memory_utilization: GPU memory utilization ratio
            max_model_len: Maximum sequence length
        """
        print(f"Loading {model_name} with vLLM...")
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,  # Required for Qwen models
        )
        
        # Sampling parameters for consistent transliteration
        self.sampling_params = SamplingParams(
            temperature=0.1,  # Low temperature for consistency
            top_p=0.9,
            max_tokens=512,
            stop=["<|endoftext|>", "<|im_end|>"],
        )
        
        print("Model loaded successfully!")
    
    def create_transliteration_prompt(self, hindi_text: str) -> str:
        """
        Create a proper prompt for Qwen model with simple transliteration
        
        Args:
            hindi_text: Input Hindi/Hinglish text
            
        Returns:
            Formatted prompt for the model
        """
        prompt = f"""<|im_start|>system
You are an expert Hindi to English transliteration assistant. Your task is to convert Hindi (Devanagari script) words to English (Roman script) using SIMPLE transliteration without any diacritical marks.

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
<|im_end|>
<|im_start|>user
Transliterate this text: {hindi_text}
<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def transliterate_batch(self, texts: List[str]) -> List[str]:
        """
        Transliterate a batch of texts using vLLM
        
        Args:
            texts: List of Hindi/Hinglish texts
            
        Returns:
            List of transliterated texts
        """
        # Create prompts for all texts
        prompts = [self.create_transliteration_prompt(text) for text in texts]
        
        # Generate responses using vLLM
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        # Extract and clean responses
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            # Clean up any extra tokens or formatting
            cleaned_text = self.clean_output(generated_text)
            results.append(cleaned_text)
        
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
        Clean up GPU memory and resources
        """
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("GPU memory cleaned up successfully")
            except Exception as e:
                print(f"Error during cleanup: {e}")
    
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
                           batch_size: int = 32,
                           text_column: str = "text") -> str:
        """
        Process the metadata CSV file and transliterate random rows
        
        Args:
            input_file: Input CSV file path
            output_file: Output CSV file path (if None, auto-generated)
            num_rows_to_transliterate: Number of random rows to transliterate
            batch_size: Batch size for vLLM processing
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

def main():
    # Configuration
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    BATCH_SIZE = 16  # Adjust based on your GPU memory
    NUM_ROWS_TO_TRANSLITERATE = 20000
    
    # File paths
    INPUT_FILE = "/media/rdp/New Volume/F5-TTS/Combined_Hindi_TTS_Hours_Limited/metadata.csv"
    OUTPUT_FILE = "/media/rdp/New Volume/F5-TTS/Combined_Hindi_TTS_Hours_Limited/metadata_transliterated.csv"
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. This will be very slow on CPU.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    # Initialize transliterator
    try:
        transliterator = VLLMQwenTransliterator(
            model_name=MODEL_NAME,
            tensor_parallel_size=1,  # Increase if you have multiple GPUs
            gpu_memory_utilization=0.6,  # Reduced to prevent memory conflicts
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have vLLM installed: pip install vllm")
        return
    
    # Test with a single example first
    test_sentences = [
        "इन तारों को लीड्स कहते हैं",
        "परीक्षा में स्लीवलेस नथनी ईयरिंग माला और साइड के बटन तक निकलवा लिए गए",
        "चश्मदीदों ने बताया कि इस हादसे के बाद पटरियों के दोनों ओर खून से सनी लाशें पड़ीं थीं",
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
    
    # Clean up GPU memory
    try:
        transliterator.cleanup()
    except Exception as e:
        print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()
