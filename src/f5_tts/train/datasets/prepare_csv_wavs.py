import concurrent.futures
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
from contextlib import contextmanager
import re
import unicodedata

sys.path.append(os.getcwd())

import argparse
import csv
import json
from pathlib import Path
import librosa
import soundfile as sf
import torchaudio
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

# Configuration constants
BATCH_SIZE = 100
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)
THREAD_NAME_PREFIX = "AudioProcessor"
CHUNK_SIZE = 50  # Reduced for audio processing
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1  # Mono

executor = None


def normalize_hindi_english_text(text):
    """
    Normalize Hindi and English text by:
    1. Converting to lowercase (for English parts)
    2. Removing extra whitespace
    3. Normalizing Unicode characters
    4. Removing special characters but keeping basic punctuation
    """
    # Normalize Unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Keep alphanumeric characters, spaces, and basic punctuation for both Hindi and English
    # This regex keeps:
    # - Latin characters (English): a-zA-Z
    # - Devanagari characters (Hindi): \u0900-\u097F
    # - Numbers: 0-9
    # - Basic punctuation: .,!?;:-'"()[]
    # - Spaces
    text = re.sub(r'[^\w\s\u0900-\u097F.,!?;:\-\'"()\[\]]+', ' ', text)
    
    # Convert English parts to lowercase while preserving Hindi
    # This is a simple approach - you might want more sophisticated language detection
    result = ""
    for char in text:
        if char.isalpha() and ord(char) < 256:  # Basic Latin characters
            result += char.lower()
        else:
            result += char
    
    # Clean up extra spaces again
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result


def create_vocab_from_text(texts):
    """Create vocabulary set from Hindi-English texts"""
    vocab_set = set()
    for text in texts:
        # Add all unique characters to vocabulary
        vocab_set.update(list(text))
    return vocab_set


def batch_normalize_texts(texts, batch_size=BATCH_SIZE):
    """Normalize a list of texts in batches."""
    normalized_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        normalized_batch = [normalize_hindi_english_text(text) for text in batch]
        normalized_texts.extend(normalized_batch)
    return normalized_texts


def convert_audio_format(input_path, output_path, target_sr=TARGET_SAMPLE_RATE, target_channels=TARGET_CHANNELS):
    """
    Convert audio to target sample rate and channel configuration using librosa and soundfile
    """
    try:
        # Load audio file
        audio, sr = librosa.load(input_path, sr=None, mono=False)
        
        # Convert to mono if needed
        if audio.ndim > 1 and target_channels == 1:
            audio = librosa.to_mono(audio)
        elif audio.ndim == 1 and target_channels > 1:
            # If input is mono but we want stereo (unlikely for TTS)
            audio = audio.reshape(1, -1)
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Ensure audio is in the right shape
        if target_channels == 1 and audio.ndim > 1:
            audio = audio.flatten()
        
        # Save the converted audio
        sf.write(output_path, audio, target_sr)
        return True
        
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False


def process_audio_file(audio_path, text, output_dir):
    """Process a single audio file: check existence, convert format, and extract duration."""
    input_path = Path(audio_path)
    if not input_path.exists():
        print(f"Audio {audio_path} not found, skipping")
        return None
    
    try:
        # Create output path with same filename but in processed directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / input_path.name
        
        # Check if file needs conversion
        needs_conversion = False
        try:
            # Check current audio properties
            info = torchaudio.info(str(input_path))
            if info.sample_rate != TARGET_SAMPLE_RATE or info.num_channels != TARGET_CHANNELS:
                needs_conversion = True
        except Exception:
            needs_conversion = True
        
        if needs_conversion:
            # Convert audio format
            if not convert_audio_format(str(input_path), str(output_path)):
                return None
            final_audio_path = str(output_path)
        else:
            # Copy file if already in correct format
            shutil.copy2(str(input_path), str(output_path))
            final_audio_path = str(output_path)
        
        # Get duration of processed audio
        audio_duration = get_audio_duration(final_audio_path)
        if audio_duration <= 0:
            raise ValueError(f"Duration {audio_duration} is non-positive.")
            
        return (final_audio_path, text, audio_duration)
        
    except Exception as e:
        print(f"Warning: Failed to process {audio_path} due to error: {e}. Skipping corrupt file.")
        return None


@contextmanager
def graceful_exit():
    """Context manager for graceful shutdown on signals"""
    def signal_handler(signum, frame):
        print("\nReceived signal to terminate. Cleaning up...")
        if executor is not None:
            print("Shutting down executor...")
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        yield
    finally:
        if executor is not None:
            executor.shutdown(wait=False)


def is_csv_wavs_format(input_dataset_dir):
    fpath = Path(input_dataset_dir)
    metadata = fpath / "metadata.csv"
    wavs = fpath / "wavs"
    return metadata.exists() and metadata.is_file() and wavs.exists() and wavs.is_dir()


def prepare_csv_wavs_dir(input_dir, output_dir, num_workers=None):
    global executor
    assert is_csv_wavs_format(input_dir), f"not csv_wavs format: {input_dir}"
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    metadata_path = input_dir / "metadata.csv"
    
    # Create processed audio directory
    processed_audio_dir = output_dir / "processed_wavs"
    processed_audio_dir.mkdir(parents=True, exist_ok=True)
    
    audio_path_text_pairs = read_audio_text_pairs(metadata_path.as_posix())
    total_files = len(audio_path_text_pairs)

    worker_count = num_workers if num_workers is not None else min(MAX_WORKERS, total_files)
    print(f"\nProcessing {total_files} audio files using {worker_count} workers...")
    print(f"Converting audio to {TARGET_SAMPLE_RATE}Hz, {TARGET_CHANNELS} channel(s)...")

    with graceful_exit():
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix=THREAD_NAME_PREFIX
        ) as exec:
            executor = exec
            results = []

            # Create a single progress bar for the entire process
            total_chunks = (total_files + CHUNK_SIZE - 1) // CHUNK_SIZE
            
            with tqdm(total=total_files, desc="Processing audio files", unit="files") as pbar:
                for i in range(0, len(audio_path_text_pairs), CHUNK_SIZE):
                    chunk = audio_path_text_pairs[i : i + CHUNK_SIZE]
                    chunk_futures = [
                        executor.submit(process_audio_file, pair[0], pair[1], processed_audio_dir) 
                        for pair in chunk
                    ]

                    # Process futures without nested progress bars
                    for future in chunk_futures:
                        try:
                            result = future.result()
                            if result is not None:
                                results.append(result)
                            pbar.update(1)  # Update progress for each file
                        except Exception as e:
                            print(f"Error processing file: {e}")
                            pbar.update(1)  # Still update progress even on error
                    
                    # Optional: Print summary every 100 chunks (5000 files)
                    current_chunk = i // CHUNK_SIZE + 1
                    if current_chunk % 100 == 0:
                        processed_count = len(results)
                        print(f"\nProgress: {current_chunk}/{total_chunks} chunks | {processed_count} files successfully processed")

            executor = None

    processed = [res for res in results if res is not None]
    if not processed:
        raise RuntimeError("No valid audio files were processed!")

    # Batch process text normalization
    raw_texts = [item[1] for item in processed]
    normalized_texts = batch_normalize_texts(raw_texts, batch_size=BATCH_SIZE)

    # Create vocabulary from normalized texts
    vocab_set = create_vocab_from_text(normalized_texts)

    # Prepare final results
    sub_result = []
    durations = []

    for (audio_path, _, duration), norm_text in zip(processed, normalized_texts):
        sub_result.append({
            "audio_path": audio_path, 
            "text": norm_text, 
            "duration": duration
        })
        durations.append(duration)

    return sub_result, durations, vocab_set


def get_audio_duration(audio_path, timeout=10):
    """Get the duration of an audio file in seconds using ffmpeg's ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
            text=True, check=True, timeout=timeout
        )
        duration_str = result.stdout.strip()
        if duration_str:
            return float(duration_str)
        raise ValueError("Empty duration string from ffprobe.")
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
        print(f"Warning: ffprobe failed for {audio_path} with error: {e}. Falling back to torchaudio.")
        try:
            audio, sample_rate = torchaudio.load(audio_path)
            return audio.shape[1] / sample_rate
        except Exception as e:
            raise RuntimeError(f"Both ffprobe and torchaudio failed for {audio_path}: {e}")


def read_audio_text_pairs(csv_file_path):
    audio_text_pairs = []
    parent = Path(csv_file_path).parent

    with open(csv_file_path, mode="r", newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile, delimiter="|")
        next(reader)  # Skip the header row
        
        # Debug: Print first few rows to understand the format
        row_count = 0
        for row in reader:
            if len(row) >= 2:
                audio_file = row[0].strip()
                text = row[1].strip()
                
                # Debug output for first 3 files
                if row_count < 3:
                    print(f"Debug - Row {row_count}: audio_file='{audio_file}', text='{text[:50]}...'")
                
                # Check if audio_file already contains path info or is just filename
                if "/" in audio_file or "\\" in audio_file:
                    # If it contains path separators, use as-is relative to parent
                    audio_file_path = parent / audio_file
                else:
                    # If it's just a filename, assume it's in the wavs folder
                    audio_file_path = parent / "wavs" / audio_file
                
                # Debug output for first 3 files
                if row_count < 3:
                    print(f"Debug - Constructed path: {audio_file_path}")
                    print(f"Debug - File exists: {audio_file_path.exists()}")
                    
                audio_text_pairs.append((audio_file_path.as_posix(), text))
                row_count += 1

    print(f"Debug - Total rows processed: {row_count}")
    return audio_text_pairs


def save_prepped_dataset(out_dir, result, duration_list, text_vocab_set):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving dataset to {out_dir} ...")

    # Save main dataset
    raw_arrow_path = out_dir / "raw.arrow"
    with ArrowWriter(path=raw_arrow_path.as_posix()) as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)
        writer.finalize()

    # Save durations
    dur_json_path = out_dir / "duration.json"
    with open(dur_json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # Save vocabulary
    vocab_path = out_dir / "vocab.txt"
    with open(vocab_path.as_posix(), "w", encoding="utf-8") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    dataset_name = out_dir.stem
    print(f"\nDataset: {dataset_name}")
    print(f"Sample count: {len(result)}")
    print(f"Vocab size: {len(text_vocab_set)}")
    print(f"Total duration: {sum(duration_list) / 3600:.2f} hours")
    print(f"Audio format: {TARGET_SAMPLE_RATE}Hz, {TARGET_CHANNELS} channel(s)")


def prepare_and_save_set(inp_dir, out_dir, num_workers=None):
    sub_result, durations, vocab_set = prepare_csv_wavs_dir(inp_dir, out_dir, num_workers=num_workers)
    save_prepped_dataset(out_dir, sub_result, durations, vocab_set)


def cli():
    try:
        # Check dependencies
        missing_deps = []
        try:
            import librosa
            import soundfile
        except ImportError as e:
            missing_deps.append(str(e))
        
        if missing_deps:
            print("Missing dependencies. Please install:")
            print("pip install librosa soundfile")
            for dep in missing_deps:
                print(f"  - {dep}")
            sys.exit(1)

        if shutil.which("ffprobe") is None:
            print("Warning: ffprobe not available. Duration extraction will use torchaudio (slower).")

        parser = argparse.ArgumentParser(
            description="Prepare Hindi-English TTS dataset with audio format standardization.",
            epilog="""
Examples:
    # Basic usage:
    python prepare_hindi_english.py /input/dataset/path /output/dataset/path
    
    # With custom worker count:
    python prepare_hindi_english.py /input/dataset/path /output/dataset/path --workers 4
    
    # For your specific path:
    python prepare_hindi_english.py "/media/rdp/New Volume/F5-TTS/Combined_Hindi_TTS_Raw_Data" "/path/to/output"
            """,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument("inp_dir", type=str, help="Input directory containing metadata.csv and wavs folder")
        parser.add_argument("out_dir", type=str, help="Output directory for processed dataset")
        parser.add_argument("--workers", type=int, help=f"Number of worker threads (default: {MAX_WORKERS})")
        
        args = parser.parse_args()

        print(f"Input directory: {args.inp_dir}")
        print(f"Output directory: {args.out_dir}")
        print(f"Target audio format: {TARGET_SAMPLE_RATE}Hz, {'mono' if TARGET_CHANNELS == 1 else 'stereo'}")
        
        prepare_and_save_set(args.inp_dir, args.out_dir, num_workers=args.workers)
        print("\nProcessing completed successfully!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Cleaning up...")
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
