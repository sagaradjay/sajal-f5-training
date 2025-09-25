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

# Dataset configurations
DATASETS = [
    {
        "name": "SPRINGLab/IndicTTS-Hindi",
        "split": "train",
        "audio_column": "audio",
        "text_column": "text",
        "description": "IndicTTS Hindi dataset"
    },
    {
        "name": "skywalker290/Hindi_TTS_M", 
        "split": "train",
        "audio_column": "audio",
        "text_column": "raw_text",
        "description": "Hindi TTS Male dataset"
    },
    {
        "name": "SPRINGLab/IndicTTS-English",
        "split": "train", 
        "audio_column": "audio",
        "text_column": "text",
        "description": "IndicTTS English dataset",
        "sample_strategy": "start_end",  # Take first 20k and last 20k
        "sample_count": 20000
    },
    {
        "name": "himanshu23099/singe_speaker_hindi_audio",
        "split": "train",
        "audio_column": "audio", 
        "text_column": "text",  # We'll try common column names
        "description": "Single speaker Hindi audio dataset"
    },
    {
        "name": "ujs/hinglish",
        "split": "train",
        "audio_column": "audio",
        "text_column": "sentence",
        "description": "Hinglish audio dataset"
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

def process_local_dataset(local_path, csv_data, durations, global_idx, wavs_dir):
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

def process_additional_local_dataset(dataset_path, dataset_name, csv_data, durations, global_idx, wavs_dir):
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

def group_samples_by_speaker(csv_data, durations):
    """Group samples by speaker based on dataset source and characteristics"""
    speaker_groups = defaultdict(list)
    
    for i, (entry, duration) in enumerate(zip(csv_data, durations)):
        audio_file = entry['audio_file']
        text = entry['text']
        
        # Determine speaker group based on file patterns and content
        speaker_id = "unknown"
        
        # Local Hindi-F dataset (original dataset)
        if "audio_" in audio_file and i < 16512:  # Assuming first ~16k are from Hindi-F
            speaker_id = "hindi_f_speaker"
        
        # Additional female dataset
        elif i >= 16512 and i < 17370:  # Approximate range for female data
            if len(text.strip()) == 1:  # Single character (aksharas)
                speaker_id = "female_aksharas"
            elif any(eng_word in text.lower() for eng_word in ['the', 'and', 'is', 'to', 'in']):
                speaker_id = "female_bilingual"
            else:
                speaker_id = "female_numbers"
        
        # Additional male dataset  
        elif i >= 17370 and i < 18228:  # Approximate range for male data
            if len(text.strip()) == 1:  # Single character (aksharas)
                speaker_id = "male_aksharas"
            elif any(eng_word in text.lower() for eng_word in ['the', 'and', 'is', 'to', 'in']):
                speaker_id = "male_bilingual"
            else:
                speaker_id = "male_numbers"
        
        # Hugging Face datasets - use dataset patterns
        else:
            # Try to identify from text patterns or assume different speakers
            if len(text) < 50:  # Short texts likely from structured datasets
                speaker_id = f"hf_speaker_{i // 1000}"  # Group every 1000 samples
            else:
                speaker_id = f"hf_speaker_{i // 500}"   # Smaller groups for longer texts
        
        speaker_groups[speaker_id].append({
            'index': i,
            'audio_file': audio_file,
            'text': text,
            'duration': duration
        })
    
    return speaker_groups

def create_concatenated_samples(speaker_groups, csv_data, durations, global_idx, wavs_dir, concatenation_percentage=0.1):
    """Create concatenated longer samples from existing data"""
    print(f"\n{'='*60}")
    print("CREATING CONCATENATED LONGER SEQUENCES")
    print(f"{'='*60}")
    
    total_samples = len(csv_data)
    target_concatenated_samples = int(total_samples * concatenation_percentage)
    
    print(f"Target concatenated samples: {target_concatenated_samples} ({concatenation_percentage*100}% of {total_samples})")
    
    concatenated_data = []
    concatenated_durations = []
    created_samples = 0
    
    for speaker_id, samples in speaker_groups.items():
        if len(samples) < 2:  # Need at least 2 samples to concatenate
            continue
            
        # Filter samples that are suitable for concatenation (not too long individually)
        suitable_samples = [s for s in samples if s['duration'] <= 8.0]  # Max 8 seconds per clip
        
        if len(suitable_samples) < 2:
            continue
            
        # Calculate how many concatenated samples to create for this speaker
        speaker_target = max(1, int(target_concatenated_samples * len(suitable_samples) / total_samples))
        speaker_created = 0
        
        print(f"Processing speaker {speaker_id}: {len(suitable_samples)} suitable samples, target: {speaker_target}")
        
        # Create concatenated samples
        for _ in range(speaker_target * 3):  # Try more times to account for failures
            if speaker_created >= speaker_target:
                break
                
            # Randomly select 2-4 samples to concatenate
            num_to_concat = random.randint(2, min(4, len(suitable_samples)))
            selected_samples = random.sample(suitable_samples, num_to_concat)
            
            # Sort by original index to maintain some order
            selected_samples.sort(key=lambda x: x['index'])
            
            # Calculate total duration
            total_duration = sum(s['duration'] for s in selected_samples)
            
            # Skip if too long (F5-TTS limit is 30 seconds)
            if total_duration > 25.0:  # Leave some margin
                continue
            
            try:
                # Load and concatenate audio files
                concatenated_audio = []
                concatenated_text = []
                
                for sample in selected_samples:
                    # Load the already processed audio file
                    audio_path = wavs_dir / sample['audio_file'].replace('wavs/', '')
                    if not audio_path.exists():
                        break
                        
                    audio_data, sample_rate = librosa.load(str(audio_path), sr=None)
                    concatenated_audio.append(audio_data)
                    concatenated_text.append(sample['text'].strip())
                
                if len(concatenated_audio) != len(selected_samples):
                    continue  # Some files were missing
                
                # Concatenate audio with small silence gaps
                silence_gap = np.zeros(int(sample_rate * 0.1))  # 0.1 second silence
                final_audio = concatenated_audio[0]
                
                for audio_segment in concatenated_audio[1:]:
                    final_audio = np.concatenate([final_audio, silence_gap, audio_segment])
                
                # Concatenate text with spaces
                final_text = ' '.join(concatenated_text)
                
                # Verify final duration
                final_duration = len(final_audio) / sample_rate
                if final_duration < 1.0 or final_duration > 25.0:
                    continue
                
                # Save concatenated audio
                concat_filename = f"audio_{global_idx:06d}.wav"
                concat_path = wavs_dir / concat_filename
                sf.write(str(concat_path), final_audio, sample_rate)
                
                # Add to data
                concatenated_data.append({
                    'audio_file': f"wavs/{concat_filename}",
                    'text': final_text
                })
                concatenated_durations.append(final_duration)
                
                speaker_created += 1
                created_samples += 1
                global_idx += 1
                
            except Exception as e:
                print(f"    Error creating concatenated sample: {e}")
                continue
        
        print(f"  Created {speaker_created} concatenated samples for {speaker_id}")
    
    print(f"\nTotal concatenated samples created: {created_samples}")
    return concatenated_data, concatenated_durations, global_idx

def main():
    # Configuration
    output_dir = "Combined_Hindi_TTS"  # New output directory name
    local_dataset_path = "/media/rdp/New Volume/F5-TTS/Hindi-F"  # Path to local dataset
    additional_female_path = "/media/rdp/New Volume/F5-TTS/additional_data_female"
    additional_male_path = "/media/rdp/New Volume/F5-TTS/additional_data_male"
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    wavs_dir = output_path / "wavs"
    wavs_dir.mkdir(exist_ok=True)
    
    # Prepare CSV data
    csv_data = []
    durations = []
    global_idx = 0
    total_valid_samples = 0
    
    print("Starting multi-dataset processing...")
    
    # First, process the main local dataset
    print(f"\n{'='*60}")
    print("Processing MAIN LOCAL DATASET (Hindi-F)")
    print(f"{'='*60}")
    local_valid_samples, global_idx = process_local_dataset(
        local_dataset_path, csv_data, durations, global_idx, wavs_dir
    )
    print(f"Valid samples from main local dataset: {local_valid_samples}")
    total_valid_samples += local_valid_samples
    
    # Process additional female dataset
    print(f"\n{'='*60}")
    print("Processing ADDITIONAL FEMALE DATASET")
    print(f"{'='*60}")
    female_valid_samples, global_idx = process_additional_local_dataset(
        additional_female_path, "Female", csv_data, durations, global_idx, wavs_dir
    )
    print(f"Valid samples from female dataset: {female_valid_samples}")
    total_valid_samples += female_valid_samples
    
    # Process additional male dataset
    print(f"\n{'='*60}")
    print("Processing ADDITIONAL MALE DATASET")
    print(f"{'='*60}")
    male_valid_samples, global_idx = process_additional_local_dataset(
        additional_male_path, "Male", csv_data, durations, global_idx, wavs_dir
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
                
                # Keep duration for statistics
                durations.append(duration)
                dataset_valid_samples += 1
                global_idx += 1
        
        print(f"Valid samples from {config['name']}: {dataset_valid_samples}")
        total_valid_samples += dataset_valid_samples
    
    # Analyze duration distribution
    analyze_duration_distribution(durations)
    
    # Group samples by speaker
    print(f"\n{'='*60}")
    print("GROUPING SAMPLES BY SPEAKER")
    print(f"{'='*60}")
    speaker_groups = group_samples_by_speaker(csv_data, durations)
    
    print(f"Found {len(speaker_groups)} speaker groups:")
    for speaker_id, samples in speaker_groups.items():
        avg_duration = np.mean([s['duration'] for s in samples])
        print(f"  {speaker_id}: {len(samples)} samples, avg duration: {avg_duration:.2f}s")
    
    # Create concatenated samples
    concatenated_data, concatenated_durations, global_idx = create_concatenated_samples(
        speaker_groups, csv_data, durations, global_idx, wavs_dir, concatenation_percentage=0.1
    )
    
    # Add concatenated samples to main dataset
    csv_data.extend(concatenated_data)
    durations.extend(concatenated_durations)
    total_valid_samples += len(concatenated_data)
    
    print(f"\nFinal dataset size: {len(csv_data)} samples")
    
    # Write metadata.csv with pipe delimiter
    csv_path = output_path / "metadata.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['audio_file', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='|')
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)
    
    print(f"\n{'='*60}")
    print(f"Multi-dataset conversion completed!")
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
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 