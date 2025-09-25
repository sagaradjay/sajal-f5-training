import concurrent.futures
import multiprocessing
import os
import shutil
import signal
import subprocess  # For invoking ffprobe
import sys
from contextlib import contextmanager


sys.path.append(os.getcwd())

import argparse
import csv
import json
from importlib.resources import files
from pathlib import Path

import torchaudio
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from f5_tts.model.utils import convert_char_to_pinyin


PRETRAINED_VOCAB_PATH = files("f5_tts").joinpath("../../data/Emilia_ZH_EN_pinyin/vocab.txt")


def is_csv_wavs_format(input_dataset_dir):
    fpath = Path(input_dataset_dir)
    metadata = fpath / "metadata.csv"
    wavs = fpath / "wavs"
    return metadata.exists() and metadata.is_file() and wavs.exists() and wavs.is_dir()


# Configuration constants
BATCH_SIZE = 100  # Batch size for text conversion
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
THREAD_NAME_PREFIX = "AudioProcessor"
CHUNK_SIZE = 100  # Number of files to process per worker batch
TARGET_SAMPLE_RATE = 24000  # Target sample rate for preprocessing

executor = None  # Global executor for cleanup


@contextmanager
def graceful_exit():
    """Context manager for graceful shutdown on signals"""

    def signal_handler(signum, frame):
        print("\nReceived signal to terminate. Cleaning up...")
        if executor is not None:
            print("Shutting down executor...")
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        yield
    finally:
        if executor is not None:
            executor.shutdown(wait=False)


def preprocess_single_audio(args):
    """Preprocess a single audio file to mono 24kHz"""
    input_path, output_path = args
    
    try:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(input_path)
        
        # Check original properties
        original_channels = waveform.shape[0]
        original_sample_rate = sample_rate
        needs_conversion = False
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            needs_conversion = True
            
        # Resample if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
            needs_conversion = True
            
        # Save the processed audio
        torchaudio.save(output_path, waveform, TARGET_SAMPLE_RATE)
        
        return {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'original_channels': original_channels,
            'original_sample_rate': original_sample_rate,
            'final_channels': 1,
            'final_sample_rate': TARGET_SAMPLE_RATE,
            'needed_conversion': needs_conversion,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'original_channels': None,
            'original_sample_rate': None,
            'final_channels': None,
            'final_sample_rate': None,
            'needed_conversion': None,
            'success': False,
            'error': str(e)
        }


def preprocess_audio_directory(input_dir, num_workers=None):
    """Preprocess all audio files in the wavs directory to mono 24kHz"""
    input_dir = Path(input_dir)
    wavs_dir = input_dir / "wavs"
    wavs_24khz_dir = input_dir / "wavs_24khz"
    
    if not wavs_dir.exists():
        raise ValueError(f"wavs directory not found: {wavs_dir}")
    
    # Create output directory
    wavs_24khz_dir.mkdir(exist_ok=True)
    
    # Find all audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(wavs_dir.glob(f"*{ext}"))
    
    if not audio_files:
        raise ValueError(f"No audio files found in {wavs_dir}")
    
    print(f"Found {len(audio_files)} audio files to preprocess")
    print(f"Converting to mono {TARGET_SAMPLE_RATE}Hz and saving to {wavs_24khz_dir}")
    
    # Prepare arguments for multiprocessing
    args_list = []
    for audio_file in audio_files:
        output_file = wavs_24khz_dir / f"{audio_file.stem}.wav"  # Always save as .wav
        args_list.append((audio_file, output_file))
    
    # Use provided worker count or calculate optimal number
    worker_count = num_workers if num_workers is not None else min(MAX_WORKERS, len(audio_files))
    
    # Process files with multiprocessing
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        # Process in chunks to show progress
        for i in range(0, len(args_list), CHUNK_SIZE):
            chunk = args_list[i:i + CHUNK_SIZE]
            chunk_futures = [executor.submit(preprocess_single_audio, args) for args in chunk]
            
            for future in tqdm(
                chunk_futures,
                desc=f"Processing chunk {i // CHUNK_SIZE + 1}/{(len(args_list) + CHUNK_SIZE - 1) // CHUNK_SIZE}",
                total=len(chunk)
            ):
                result = future.result()
                results.append(result)
    
    # Analyze results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    needed_conversion = [r for r in successful if r['needed_conversion']]
    already_correct = [r for r in successful if not r['needed_conversion']]
    
    # Print statistics
    print(f"\n=== Audio Preprocessing Statistics ===")
    print(f"Total files processed: {len(results)}")
    print(f"Successfully converted: {len(successful)}")
    print(f"Failed conversions: {len(failed)}")
    print(f"Files that needed conversion: {len(needed_conversion)}")
    print(f"Files already in correct format: {len(already_correct)}")
    
    if needed_conversion:
        print(f"\nConversion details:")
        sample_rates = {}
        channels = {}
        for r in needed_conversion:
            sr = r['original_sample_rate']
            ch = r['original_channels']
            sample_rates[sr] = sample_rates.get(sr, 0) + 1
            channels[ch] = channels.get(ch, 0) + 1
        
        print(f"Original sample rates: {dict(sample_rates)}")
        print(f"Original channel counts: {dict(channels)}")
    
    if failed:
        print(f"\nFailed files:")
        for r in failed[:10]:  # Show first 10 failed files
            print(f"  {r['input_path']}: {r['error']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    # Update metadata.csv to point to new wavs_24khz directory
    metadata_path = input_dir / "metadata.csv"
    if metadata_path.exists():
        new_metadata_path = input_dir / "metadata_24khz.csv"
        print(f"\nUpdating metadata file: {new_metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8-sig') as infile, \
             open(new_metadata_path, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.reader(infile, delimiter='|')
            writer = csv.writer(outfile, delimiter='|')
            
            # Copy header
            header = next(reader)
            writer.writerow(header)
            
            # Update paths in data rows
            for row in reader:
                if len(row) >= 2:
                    original_audio_path = row[0].strip()
                    # Extract filename and change extension to .wav, point to wavs_24khz
                    filename = Path(original_audio_path).stem + '.wav'
                    new_audio_path = f"wavs_24khz/{filename}"
                    
                    new_row = [new_audio_path] + row[1:]
                    writer.writerow(new_row)
        
        print(f"Updated metadata saved as: {new_metadata_path}")
        return new_metadata_path
    else:
        print("Warning: metadata.csv not found, cannot update paths")
        return None
    
    return results


def process_audio_file(audio_path, text, language):
    """Process a single audio file by checking its existence and extracting duration."""
    if not Path(audio_path).exists():
        print(f"audio {audio_path} not found, skipping")
        return None
    try:
        audio_duration = get_audio_duration(audio_path)
        if audio_duration <= 0:
            raise ValueError(f"Duration {audio_duration} is non-positive.")
        return (audio_path, text, audio_duration)
    except Exception as e:
        print(f"Warning: Failed to process {audio_path} due to error: {e}. Skipping corrupt file.")
        return None


def normalize_hindi_text(text):
    """Normalize Hindi text for better consistency"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Basic punctuation normalization
    text = text.replace('।', '.')  # Replace devanagari danda with period
    text = text.replace('॥', '..')  # Replace double danda
    
    # Remove or normalize other problematic characters if needed
    # Add more normalization rules as you discover issues
    
    return text


def batch_convert_texts(texts, language, polyphone=True, batch_size=BATCH_SIZE):
    """Convert texts based on language - pinyin for Chinese, keep original for others."""
    if language.lower() == "chinese":
        converted_texts = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            converted_batch = convert_char_to_pinyin(batch, polyphone=polyphone)
            converted_texts.extend(converted_batch)
        return converted_texts
    elif language.lower() in ["hindi", "hindi1"]:
        # For Hindi, normalize but don't convert to pinyin
        return [normalize_hindi_text(text) for text in texts]
    else:
        # For other languages, return as-is
        return texts


def prepare_csv_wavs_dir(input_dir, language="chinese", num_workers=None, use_preprocessed=False):
    global executor
    assert is_csv_wavs_format(input_dir), f"not csv_wavs format: {input_dir}"
    input_dir = Path(input_dir)
    
    # Choose which metadata file to use
    if use_preprocessed:
        metadata_path = input_dir / "metadata_24khz.csv"
        if not metadata_path.exists():
            raise ValueError("Preprocessed metadata file not found. Run preprocessing first.")
    else:
        metadata_path = input_dir / "metadata.csv"
    
    audio_path_text_pairs = read_audio_text_pairs(metadata_path.as_posix())

    polyphone = True
    total_files = len(audio_path_text_pairs)

    # Use provided worker count or calculate optimal number
    worker_count = num_workers if num_workers is not None else min(MAX_WORKERS, total_files)
    print(f"\nProcessing {total_files} audio files using {worker_count} workers...")
    print(f"Language: {language}")
    print(f"Using preprocessed audio: {use_preprocessed}")

    with graceful_exit():
        # Initialize thread pool with optimized settings
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix=THREAD_NAME_PREFIX
        ) as exec:
            executor = exec
            results = []

            # Process files in chunks for better efficiency
            for i in range(0, len(audio_path_text_pairs), CHUNK_SIZE):
                chunk = audio_path_text_pairs[i : i + CHUNK_SIZE]
                # Submit futures in order
                chunk_futures = [executor.submit(process_audio_file, pair[0], pair[1], language) for pair in chunk]

                # Iterate over futures in the original submission order to preserve ordering
                for future in tqdm(
                    chunk_futures,
                    total=len(chunk),
                    desc=f"Processing chunk {i // CHUNK_SIZE + 1}/{(total_files + CHUNK_SIZE - 1) // CHUNK_SIZE}",
                ):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        print(f"Error processing file: {e}")

            executor = None

    # Filter out failed results
    processed = [res for res in results if res is not None]
    if not processed:
        raise RuntimeError("No valid audio files were processed!")

    # Batch process text conversion based on language
    raw_texts = [item[1] for item in processed]
    converted_texts = batch_convert_texts(raw_texts, language, polyphone, batch_size=BATCH_SIZE)

    # Prepare final results and build vocabulary
    sub_result = []
    durations = []
    vocab_set = set()

    for (audio_path, _, duration), conv_text in zip(processed, converted_texts):
        sub_result.append({"audio_path": audio_path, "text": conv_text, "duration": duration})
        durations.append(duration)
        
        # Add characters to vocabulary set
        # For Hindi, this will include Devanagari characters
        # For Chinese with pinyin, this will include pinyin characters
        vocab_set.update(list(conv_text))

    print(f"\nVocabulary preview (first 20 characters): {sorted(list(vocab_set))[:20]}")
    print(f"Total unique characters in dataset: {len(vocab_set)}")

    return sub_result, durations, vocab_set


def get_audio_duration(audio_path, timeout=5):
    """
    Get the duration of an audio file in seconds using ffmpeg's ffprobe.
    Falls back to torchaudio.load() if ffprobe fails.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=timeout
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
        for row in reader:
            if len(row) >= 2:
                audio_file = row[0].strip()  # First column: audio file path
                text = row[1].strip()  # Second column: text
                audio_file_path = parent / audio_file
                audio_text_pairs.append((audio_file_path.as_posix(), text))

    return audio_text_pairs


def get_comprehensive_vocab_set(language):
    """Get a comprehensive character set for the specified language - FIXED to match actual dataset"""
    base_vocab = set()
    
    # English letters
    english_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    
    # English numbers
    english_numbers = set('0123456789')
    
    # Common punctuation - CORRECTED to match your dataset analysis
    punctuation = set('!"#$%&\'()*+,-./:;<=>?@\\^_{|}~""''–—…')
    
    # Space and common whitespace - CORRECTED
    whitespace = set(' \t\n')  # Only basic whitespace
    
    # Additional special characters - CORRECTED to match your dataset
    special_chars = set("°ºâñāīśū˜λμπφω€™•")
    
    # Always include these base characters
    base_vocab.update(english_letters)
    base_vocab.update(english_numbers)
    base_vocab.update(punctuation)
    base_vocab.update(whitespace)
    base_vocab.update(special_chars)
    
    if language.lower() in ["hindi", "hindi1"]:  # Support both hindi and hindi1
        # Hindi numbers (Devanagari numerals)
        hindi_numbers = set('०१२३४५६७८९')
        
        # Hindi vowels - CORRECTED to match your dataset (removed ॠ, ऌ, ॡ that aren't in your data)
        hindi_vowels = set('अआइईउऊऋऍऎएऐऑऒओऔ')
        
        # Hindi consonants - CORRECTED to match your dataset
        hindi_consonants = set('कखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह')
        
        # Hindi vowel diacritics (matras) - FIXED: Remove Bengali chars, only Hindi matras
        hindi_matras = set('ािीुूृॄॅॆेैॉॊोौ्')  # REMOVED Bengali chars ী, ু, ূ, ৃ, ৄ, ে, ৈ, ো, ৌ, ্
        
        # Hindi special characters - CORRECTED to match your dataset
        hindi_special = set('ंःँ़्ॐ')  # Simplified to commonly used chars
        
        # Hindi extended characters (commonly used)
        hindi_extended = set('क़ख़ग़ज़ड़ढ़फ़य़')
        
        # Hindi punctuation
        hindi_punctuation = set('।॥')
        
        # Hindi additional Devanagari numbers and symbols - from your analysis
        hindi_additional = set('॰ॲ')
        
        # Zero-width characters (used in Devanagari) - CORRECTED
        hindi_zw = set('\u200c\u200d\u200e')  # Zero-width non-joiner, joiner, and LTR mark
        
        # Additional characters from comprehensive vocab (transliteration marks)
        additional_devanagari = set('ḍṅṇṛṣṭ')
        
        base_vocab.update(hindi_numbers)
        base_vocab.update(hindi_vowels)
        base_vocab.update(hindi_consonants)
        base_vocab.update(hindi_matras)
        base_vocab.update(hindi_special)
        base_vocab.update(hindi_extended)
        base_vocab.update(hindi_punctuation)
        base_vocab.update(hindi_additional)
        base_vocab.update(hindi_zw)
        base_vocab.update(additional_devanagari)
    
    return base_vocab


def save_prepped_dataset(out_dir, result, duration_list, text_vocab_set, is_finetune, language):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving to {out_dir} ...")

    raw_arrow_path = out_dir / "raw.arrow"
    with ArrowWriter(path=raw_arrow_path.as_posix()) as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)
        writer.finalize()

    # Save durations to JSON
    dur_json_path = out_dir / "duration.json"
    with open(dur_json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # Handle vocab file based on language and finetune flag
    voca_out_path = out_dir / "vocab.txt"
    if is_finetune and language.lower() == "chinese":
        # For Chinese fine-tuning, use pretrained vocab
        file_vocab_finetune = PRETRAINED_VOCAB_PATH.as_posix()
        shutil.copy2(file_vocab_finetune, voca_out_path)
        print("Using pretrained Chinese vocab for fine-tuning")
    else:
        # For Hindi or training from scratch, create comprehensive vocab
        comprehensive_vocab = get_comprehensive_vocab_set(language)
        
        # Merge with characters found in dataset
        final_vocab_set = comprehensive_vocab.union(text_vocab_set)
        
        # Remove any unwanted characters (optional filtering)
        final_vocab_set = {char for char in final_vocab_set if char.isprintable() or char in ' \t\n\u200c\u200d\u200e\u2009'}
        
        # CRITICAL FIX: Remove empty strings and ensure space is at index 0
        final_vocab_set = {char for char in final_vocab_set if char != ''}  # Remove empty strings
        
        # Convert to sorted list for proper ordering
        vocab_list = sorted(final_vocab_set)
        
        # Ensure space character is at index 0 (required by tokenizer)
        if ' ' in vocab_list:
            vocab_list.remove(' ')  # Remove space from its current position
            vocab_list.insert(0, ' ')  # Insert space at index 0
        
        # Write the properly ordered vocabulary
        with open(voca_out_path.as_posix(), "w", encoding="utf-8") as f:
            for vocab in vocab_list:
                f.write(vocab + "\n")
        
        print(f"Created comprehensive vocabulary with {len(vocab_list)} characters for {language}")
        print(f"Space character is at index 0: {'✓' if vocab_list[0] == ' ' else '✗'}")
        print(f"Dataset contributed {len(text_vocab_set)} unique characters")
        print(f"Base language set contributed {len(comprehensive_vocab)} characters")

    dataset_name = out_dir.stem
    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, total vocab size: {len(final_vocab_set) if 'final_vocab_set' in locals() else 'N/A'}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")


def prepare_and_save_set(inp_dir, out_dir, language="chinese", is_finetune: bool = True, num_workers: int = None, preprocess_audio: bool = False):
    if is_finetune and language.lower() == "chinese":
        assert PRETRAINED_VOCAB_PATH.exists(), f"pretrained vocab.txt not found: {PRETRAINED_VOCAB_PATH}"
    
    # Preprocess audio if requested
    if preprocess_audio:
        print("=== Audio Preprocessing Phase ===")
        preprocess_audio_directory(inp_dir, num_workers)
        print("\n=== Dataset Processing Phase ===")
        sub_result, durations, vocab_set = prepare_csv_wavs_dir(inp_dir, language=language, num_workers=num_workers, use_preprocessed=True)
    else:
        sub_result, durations, vocab_set = prepare_csv_wavs_dir(inp_dir, language=language, num_workers=num_workers, use_preprocessed=False)
    
    save_prepped_dataset(out_dir, sub_result, durations, vocab_set, is_finetune, language)


def cli():
    try:
        # Import torch here to avoid issues if not available during argument parsing
        import torch
        
        # Before processing, check if ffprobe is available.
        if shutil.which("ffprobe") is None:
            print(
                "Warning: ffprobe is not available. Duration extraction will rely on torchaudio (which may be slower)."
            )

        # Usage examples in help text
        parser = argparse.ArgumentParser(
            description="Prepare and save dataset for different languages with optional audio preprocessing.",
            epilog="""
Examples:
    # For Hindi fine-tuning with audio preprocessing:
    python prepare_csv_wavs.py /input/dataset/path /output/dataset/path --language hindi --preprocess-audio
    
    # For Hindi training from scratch with preprocessing:
    python prepare_csv_wavs.py /input/dataset/path /output/dataset/path --language hindi --pretrain --preprocess-audio
    
    # For Chinese fine-tuning (default):
    python prepare_csv_wavs.py /input/dataset/path /output/dataset/path --language chinese
    
    # With custom worker count:
    python prepare_csv_wavs.py /input/dataset/path /output/dataset/path --language hindi --workers 4 --preprocess-audio
            """,
        )
        parser.add_argument("inp_dir", type=str, help="Input directory containing the data.")
        parser.add_argument("out_dir", type=str, help="Output directory to save the prepared data.")
        parser.add_argument("--language", type=str, default="chinese", 
                           choices=["chinese", "hindi", "hindi1", "english", "other"],
                           help="Language of the dataset (affects text processing)")
        parser.add_argument("--pretrain", action="store_true", help="Enable for new pretrain, otherwise is a fine-tune")
        parser.add_argument("--workers", type=int, help=f"Number of worker threads (default: {MAX_WORKERS})")
        parser.add_argument("--preprocess-audio", action="store_true", 
                           help="Preprocess audio files to mono 24kHz before processing")
        args = parser.parse_args()

        prepare_and_save_set(args.inp_dir, args.out_dir, 
                           language=args.language, 
                           is_finetune=not args.pretrain, 
                           num_workers=args.workers,
                           preprocess_audio=args.preprocess_audio)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Cleaning up...")
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()