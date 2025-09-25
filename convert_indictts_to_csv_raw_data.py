import os
import csv
from pathlib import Path
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm
import librosa
import numpy as np
import glob
import re
import random
from collections import defaultdict

# Dataset configurations - MODIFIED to include sample limits
DATASETS = [
    {
        "name": "SPRINGLab/IndicTTS-Hindi",
        "split": "train",
        "audio_column": "audio",
        "text_column": "text",
        "description": "IndicTTS Hindi dataset",
        "single_speaker": False  # Multi-speaker dataset
    },
    {
        "name": "skywalker290/Hindi_TTS_M", 
        "split": "train",
        "audio_column": "audio",
        "text_column": "raw_text",
        "description": "Hindi TTS Male dataset",
        "single_speaker": True  # Single speaker
    },
    {
        "name": "SPRINGLab/IndicTTS-English",
        "split": "train", 
        "audio_column": "audio",
        "text_column": "text",
        "description": "IndicTTS English dataset",
        "sample_strategy": "start_end",  # Take first 20k and last 20k
        "sample_count": 20000,
        "single_speaker": False  # Multi-speaker dataset
    },
    {
        "name": "himanshu23099/singe_speaker_hindi_audio",
        "split": "train",
        "audio_column": "audio", 
        "text_column": "text",
        "description": "Single speaker Hindi audio dataset",
        "single_speaker": True  # Single speaker
    },
    {
        "name": "ujs/hinglish",
        "split": "train",
        "audio_column": "audio",
        "text_column": "sentence",
        "description": "Hinglish audio dataset",
        "single_speaker": True,  # Single speaker
        "sample_strategy": "random",  # Random sampling
        "sample_count": 30000  # Cap to 30k samples
    }
]

def load_dataset_safely(dataset_config):
    """Load a dataset with error handling"""
    try:
        print(f"Loading {dataset_config['description']}...")
        dataset = load_dataset(dataset_config["name"], split=dataset_config["split"])
        
        # Disable automatic audio decoding to avoid TorchCodec issues
        if dataset_config["audio_column"] in dataset.column_names:
            dataset = dataset.cast_column(dataset_config["audio_column"], Audio(decode=False))
        
        print(f"Successfully loaded {len(dataset)} samples from {dataset_config['name']}")
        return dataset, dataset_config
    except Exception as e:
        print(f"Failed to load {dataset_config['name']}: {e}")
        return None, None

def extract_text_from_sample(sample, text_column, dataset_name):
    """Extract text from sample, handling different column names and structures"""
    try:
        # Try the specified text column first
        if text_column in sample and sample[text_column] is not None:
            text = sample[text_column]
            if isinstance(text, str) and text.strip():
                return text.strip()
        
        # Try common alternative column names
        alternative_columns = ["text", "raw_text", "transcript", "transcription", "sentence"]
        for col in alternative_columns:
            if col in sample and sample[col] is not None:
                text = sample[col]
                if isinstance(text, str) and text.strip():
                    return text.strip()
        
        return None
    except Exception as e:
        print(f"Error extracting text from {dataset_name}: {e}")
        return None

def process_audio_sample(sample, audio_column, text_column, dataset_name, idx, global_idx):
    """Process a single audio sample"""
    try:
        # Extract text
        text = extract_text_from_sample(sample, text_column, dataset_name)
        if not text:
            return None, None, None
        
        # Extract audio data
        if audio_column not in sample:
            return None, None, None
            
        audio_info = sample[audio_column]
        
        # Load audio from bytes using librosa
        if "bytes" in audio_info and audio_info["bytes"] is not None:
            import io
            audio_data, sample_rate = librosa.load(io.BytesIO(audio_info["bytes"]), sr=None)
        elif "path" in audio_info and audio_info["path"] is not None:
            # Load from file path
            audio_data, sample_rate = librosa.load(audio_info["path"], sr=None)
        else:
            return None, None, None
        
        # Ensure audio_data is numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        # Calculate duration
        duration = len(audio_data) / sample_rate
        
        # Filter by duration (F5-TTS requirements: 0.4-30 seconds)
        if duration < 0.4 or duration > 30:
            return None, None, None
        
        return audio_data, sample_rate, text, duration
        
    except Exception as e:
        print(f"Error processing sample {idx} from {dataset_name}: {e}")
        return None, None, None, None

def process_local_dataset(local_path, csv_data, durations, global_idx, wavs_dir, dataset_metadata):
    """Process local dataset with wav and txt files"""
    local_path = Path(local_path)
    wav_dir = local_path / "wav"
    txt_dir = local_path / "txt"
    
    if not wav_dir.exists() or not txt_dir.exists():
        print(f"Local dataset directories not found: {wav_dir} or {txt_dir}")
        return 0, global_idx
    
    # Get all wav files
    wav_files = list(wav_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} local audio files")
    
    valid_samples = 0
    
    for wav_file in tqdm(wav_files, desc="Processing local dataset"):
        try:
            # Find corresponding text file
            txt_file = txt_dir / f"{wav_file.stem}.txt"
            
            if not txt_file.exists():
                continue
            
            # Read text content
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                continue
            
            # Load audio
            audio_data, sample_rate = librosa.load(str(wav_file), sr=None)
            
            # Ensure audio_data is numpy array
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            
            # Calculate duration
            duration = len(audio_data) / sample_rate
            
            # Filter by duration (F5-TTS requirements: 0.4-30 seconds)
            if duration < 0.4 or duration > 30:
                continue
            
            # Create audio filename for output
            audio_filename = f"audio_{global_idx:06d}.wav"
            audio_path = wavs_dir / audio_filename
            
            # Save audio file
            sf.write(str(audio_path), audio_data, sample_rate)
            
            # Add to CSV data
            csv_data.append({
                'audio_file': f"wavs/{audio_filename}",
                'text': text
            })
            
            # Store metadata for this sample
            dataset_metadata.append({
                'index': len(csv_data) - 1,
                'dataset': 'Hindi-F',
                'single_speaker': True  # Hindi-F is single speaker
            })
            
            # Keep duration for statistics
            durations.append(duration)
            valid_samples += 1
            global_idx += 1
            
        except Exception as e:
            print(f"Error processing local file {wav_file.name}: {e}")
            continue
    
    return valid_samples, global_idx

def parse_festival_format(text_file):
    """Parse Festival format text file: ( filename " text " )"""
    text_dict = {}
    
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Regular expression to match ( filename " text " ) format
        pattern = r'\(\s*(\w+)\s*"\s*([^"]*)\s*"\s*\)'
        matches = re.findall(pattern, content)
        
        for filename, text in matches:
            text_dict[filename] = text.strip()
            
    except Exception as e:
        print(f"Error parsing festival format file {text_file}: {e}")
    
    return text_dict

def process_additional_local_dataset(dataset_path, dataset_name, csv_data, durations, global_idx, wavs_dir, dataset_metadata):
    """Process additional local datasets with festival format"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Dataset path not found: {dataset_path}")
        return 0, global_idx
    
    print(f"Processing {dataset_name} dataset...")
    valid_samples = 0
    
    # Categories to process
    categories = ['aksharas', 'bilingual', 'numbers']
    
    for category in categories:
        category_path = dataset_path / category
        wav_dir = category_path / "wav"
        
        # Find text file (different naming conventions)
        text_files = list(category_path.glob("*.txt"))
        if not text_files:
            text_files = list(category_path.glob("*_txt"))
        
        if not text_files or not wav_dir.exists():
            print(f"  Skipping {category}: missing text file or wav directory")
            continue
            
        text_file = text_files[0]  # Take the first text file found
        
        # Parse the festival format text file
        text_dict = parse_festival_format(text_file)
        if not text_dict:
            print(f"  Skipping {category}: no valid text entries found")
            continue
        
        # Get all wav files
        wav_files = list(wav_dir.glob("*.wav"))
        print(f"  Processing {category}: {len(wav_files)} files")
        
        category_valid = 0
        
        for wav_file in tqdm(wav_files, desc=f"{dataset_name}-{category}"):
            try:
                # Get filename without extension
                file_id = wav_file.stem
                
                # Get corresponding text
                if file_id not in text_dict:
                    continue
                
                text = text_dict[file_id]
                if not text or len(text.strip()) == 0:
                    continue
                
                # Load audio
                audio_data, sample_rate = librosa.load(str(wav_file), sr=None)
                
                # Ensure audio_data is numpy array
                if not isinstance(audio_data, np.ndarray):
                    audio_data = np.array(audio_data)
                
                # Calculate duration
                duration = len(audio_data) / sample_rate
                
                # Filter by duration (F5-TTS requirements: 0.4-30 seconds)
                if duration < 0.4 or duration > 30:
                    continue
                
                # Create audio filename for output
                audio_filename = f"audio_{global_idx:06d}.wav"
                audio_path = wavs_dir / audio_filename
                
                # Save audio file
                sf.write(str(audio_path), audio_data, sample_rate)
                
                # Add to CSV data
                csv_data.append({
                    'audio_file': f"wavs/{audio_filename}",
                    'text': text
                })
                
                # Store metadata for this sample
                dataset_metadata.append({
                    'index': len(csv_data) - 1,
                    'dataset': dataset_name,
                    'single_speaker': False  # Additional datasets are multi-speaker
                })
                
                # Keep duration for statistics
                durations.append(duration)
                valid_samples += 1
                category_valid += 1
                global_idx += 1
                
            except Exception as e:
                print(f"    Error processing {wav_file.name}: {e}")
                continue
        
        print(f"    Valid samples from {category}: {category_valid}")
    
    return valid_samples, global_idx

def analyze_duration_distribution(durations):
    """Analyze the duration distribution of the dataset"""
    if not durations:
        return
    
    durations = np.array(durations)
    
    print(f"\n{'='*50}")
    print("DURATION DISTRIBUTION ANALYSIS")
    print(f"{'='*50}")
    print(f"Total samples: {len(durations)}")
    print(f"Total duration: {np.sum(durations) / 3600:.2f} hours")
    print(f"Mean duration: {np.mean(durations):.2f} seconds")
    print(f"Median duration: {np.median(durations):.2f} seconds")
    print(f"Min duration: {np.min(durations):.2f} seconds")
    print(f"Max duration: {np.max(durations):.2f} seconds")
    print(f"Std deviation: {np.std(durations):.2f} seconds")
    
    # Duration buckets
    buckets = [
        (0.0, 1.0, "Very Short"),
        (1.0, 3.0, "Short"),
        (3.0, 5.0, "Medium"),
        (5.0, 10.0, "Long"),
        (10.0, 30.0, "Very Long")
    ]
    
    print(f"\nDuration Distribution:")
    for min_dur, max_dur, label in buckets:
        count = np.sum((durations >= min_dur) & (durations < max_dur))
        percentage = (count / len(durations)) * 100
        print(f"  {label:10} ({min_dur:4.1f}-{max_dur:4.1f}s): {count:6d} samples ({percentage:5.1f}%)")

def main():
    # Configuration
    output_dir = "Combined_Hindi_TTS_Raw_Data"  # Updated output directory name
    local_dataset_path = "/media/rdp/New Volume/F5-TTS/Hindi-F"  # Path to local dataset
    additional_female_path = "/media/rdp/New Volume/F5-TTS/additional_data_female"
    additional_male_path = "/media/rdp/New Volume/F5-TTS/additional_data_male"
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    wavs_dir = output_path / "wavs"
    wavs_dir.mkdir(exist_ok=True)
    
    # Prepare CSV data and metadata tracking
    csv_data = []
    durations = []
    dataset_metadata = []  # Track which dataset each sample came from
    global_idx = 0
    total_valid_samples = 0
    
    print(f"Starting multi-dataset processing with RAW DATA ONLY...")
    print(f"NOTE: No concatenation - using raw data only")
    print(f"NOTE: ujs/hinglish dataset capped to 30k samples")
    
    # First, process the main local dataset
    print(f"\n{'='*60}")
    print("Processing MAIN LOCAL DATASET (Hindi-F)")
    print(f"{'='*60}")
    local_valid_samples, global_idx = process_local_dataset(
        local_dataset_path, csv_data, durations, global_idx, wavs_dir, dataset_metadata
    )
    print(f"Valid samples from main local dataset: {local_valid_samples}")
    total_valid_samples += local_valid_samples
    
    # Process additional female dataset
    print(f"\n{'='*60}")
    print("Processing ADDITIONAL FEMALE DATASET")
    print(f"{'='*60}")
    female_valid_samples, global_idx = process_additional_local_dataset(
        additional_female_path, "Female", csv_data, durations, global_idx, wavs_dir, dataset_metadata
    )
    print(f"Valid samples from female dataset: {female_valid_samples}")
    total_valid_samples += female_valid_samples
    
    # Process additional male dataset
    print(f"\n{'='*60}")
    print("Processing ADDITIONAL MALE DATASET")
    print(f"{'='*60}")
    male_valid_samples, global_idx = process_additional_local_dataset(
        additional_male_path, "Male", csv_data, durations, global_idx, wavs_dir, dataset_metadata
    )
    print(f"Valid samples from male dataset: {male_valid_samples}")
    total_valid_samples += male_valid_samples
    
    # Then process each Hugging Face dataset
    print(f"\n{'='*60}")
    print("Processing HUGGING FACE DATASETS")
    print(f"{'='*60}")
    
    for dataset_config in DATASETS:
        dataset, config = load_dataset_safely(dataset_config)
        if dataset is None:
            continue
        
        # Handle sampling strategy for large datasets
        sample_strategy = config.get("sample_strategy", "all")
        sample_count = config.get("sample_count", len(dataset))
        
        if sample_strategy == "start_end" and len(dataset) > sample_count * 2:
            print(f"\nProcessing {config['description']} with start_end sampling:")
            print(f"  Total samples: {len(dataset)}")
            print(f"  Taking first {sample_count} and last {sample_count} samples (total: {sample_count * 2})")
            
            # Create indices for first N and last N samples
            indices_to_process = list(range(sample_count)) + list(range(len(dataset) - sample_count, len(dataset)))
            samples_to_process = [dataset[i] for i in indices_to_process]
            
        elif sample_strategy == "random" and len(dataset) > sample_count:
            print(f"\nProcessing {config['description']} with random sampling:")
            print(f"  Total samples: {len(dataset)}")
            print(f"  Taking random {sample_count} samples")
            
            # Randomly sample the dataset
            random_indices = random.sample(range(len(dataset)), sample_count)
            samples_to_process = [dataset[i] for i in random_indices]
            
        else:
            print(f"\nProcessing {config['description']} ({len(dataset)} samples)...")
            samples_to_process = dataset
            indices_to_process = range(len(dataset))
        
        dataset_valid_samples = 0
        
        for idx, sample in enumerate(tqdm(samples_to_process, desc=f"Processing {config['name']}")):
            result = process_audio_sample(
                sample, 
                config["audio_column"], 
                config["text_column"], 
                config["name"], 
                idx, 
                global_idx
            )
            
            if result[0] is not None:  # Valid sample
                audio_data, sample_rate, text, duration = result
                
                # Create audio filename
                audio_filename = f"audio_{global_idx:06d}.wav"
                audio_path = wavs_dir / audio_filename
                
                # Save audio file
                sf.write(str(audio_path), audio_data, sample_rate)
                
                # Add to CSV data
                csv_data.append({
                    'audio_file': f"wavs/{audio_filename}",
                    'text': text
                })
                
                # Store metadata for this sample
                dataset_metadata.append({
                    'index': len(csv_data) - 1,
                    'dataset': config["name"],
                    'single_speaker': config.get("single_speaker", False)
                })
                
                # Keep duration for statistics
                durations.append(duration)
                dataset_valid_samples += 1
                global_idx += 1
        
        print(f"Valid samples from {config['name']}: {dataset_valid_samples}")
        total_valid_samples += dataset_valid_samples
    
    # Analyze duration distribution
    analyze_duration_distribution(durations)
    
    print(f"\nFinal dataset size: {len(csv_data)} samples")
    print(f"Final total duration: {sum(durations) / 3600:.2f} hours")
    
    # Write metadata.csv with pipe delimiter
    csv_path = output_path / "metadata.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['audio_file', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='|')
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)
    
    print(f"\n{'='*60}")
    print(f"Multi-dataset conversion with RAW DATA completed!")
    print(f"{'='*60}")
    print(f"Total valid samples: {total_valid_samples}")
    if durations:
        print(f"Total duration: {sum(durations) / 3600:.2f} hours")
        print(f"Average duration: {sum(durations) / len(durations):.2f} seconds")
        print(f"Min duration: {min(durations):.2f} seconds")
        print(f"Max duration: {max(durations):.2f} seconds")
    print(f"Output directory: {output_dir}")
    print(f"Structure:")
    print(f"  {output_dir}/")
    print(f"  ├── metadata.csv")
    print(f"  └── wavs/")
    print(f"      ├── audio_000000.wav")
    print(f"      ├── audio_000001.wav")
    print(f"      └── ... (total: {total_valid_samples} files)")
    print(f"\nIMPORTANT: This version uses RAW DATA ONLY:")
    print(f"  - No concatenation - all samples are original")
    print(f"  - ujs/hinglish dataset capped to 30k samples")
    print(f"  - All datasets processed as-is")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 