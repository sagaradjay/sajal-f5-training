import os
import wave
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def get_wav_duration(file_path):
    """Get duration of a WAV file in seconds."""
    try:
        with wave.open(str(file_path), 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / float(sample_rate)
            return duration
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def plot_duration_distribution(durations, folder_name):
    """Create various plots showing audio duration distribution."""
    durations_array = np.array(durations)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Audio Duration Analysis for: {folder_name}', fontsize=16, fontweight='bold')
    
    # 1. Histogram with bins
    ax1.hist(durations_array, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.set_xlabel('Duration (seconds)')
    ax1.set_ylabel('Number of Files')
    ax1.set_title('Duration Distribution (Histogram)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax2.boxplot(durations_array, vert=True, patch_artist=True, 
                boxprops=dict(facecolor='lightgreen', alpha=0.7))
    ax2.set_ylabel('Duration (seconds)')
    ax2.set_title('Duration Distribution (Box Plot)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Duration ranges bar chart
    ranges = [
        (0, 10, "0-10s"),
        (10, 30, "10-30s"), 
        (30, 60, "30s-1m"),
        (60, 180, "1-3m"),
        (180, 300, "3-5m"),
        (300, 600, "5-10m"),
        (600, float('inf'), ">10m")
    ]
    
    range_counts = []
    range_labels = []
    
    for min_dur, max_dur, label in ranges:
        if max_dur == float('inf'):
            count = sum(1 for d in durations if d >= min_dur)
        else:
            count = sum(1 for d in durations if min_dur <= d < max_dur)
        range_counts.append(count)
        range_labels.append(f"{label}\n({count} files)")
    
    bars = ax3.bar(range_labels, range_counts, color='coral', alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Number of Files')
    ax3.set_title('Files by Duration Range')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, range_counts):
        if count > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(range_counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Cumulative distribution
    sorted_durations = np.sort(durations_array)
    cumulative_percent = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations) * 100
    
    ax4.plot(sorted_durations, cumulative_percent, color='purple', linewidth=2)
    ax4.set_xlabel('Duration (seconds)')
    ax4.set_ylabel('Cumulative Percentage')
    ax4.set_title('Cumulative Duration Distribution')
    ax4.grid(True, alpha=0.3)
    
    # Add percentile lines
    percentiles = [25, 50, 75, 90]
    for p in percentiles:
        value = np.percentile(sorted_durations, p)
        ax4.axvline(x=value, color='red', linestyle='--', alpha=0.7)
        ax4.text(value, p + 5, f'{p}th: {value:.1f}s', rotation=90, 
                fontsize=8, ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    safe_folder_name = folder_name.replace('/', '_').replace('\\', '_')
    output_file = f"audio_duration_analysis_{safe_folder_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {output_file}")
    
    plt.show()

def print_duration_statistics(durations):
    """Print detailed statistics about duration distribution."""
    durations_array = np.array(durations)
    
    print(f"\n{'='*50}")
    print(f"DURATION STATISTICS:")
    print(f"{'='*50}")
    print(f"Mean duration: {np.mean(durations_array):.2f} seconds")
    print(f"Median duration: {np.median(durations_array):.2f} seconds")
    print(f"Standard deviation: {np.std(durations_array):.2f} seconds")
    print(f"Minimum duration: {np.min(durations_array):.2f} seconds")
    print(f"Maximum duration: {np.max(durations_array):.2f} seconds")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        value = np.percentile(durations_array, p)
        print(f"  {p}th percentile: {value:.2f} seconds")

def calculate_total_duration(folder_path, show_progress=True, create_plots=True):
    """Calculate total duration of all WAV files in a folder and create plots."""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Get all WAV files
    wav_files = list(folder.glob("*.wav"))
    total_files = len(wav_files)
    
    if total_files == 0:
        print("No WAV files found in the folder.")
        return
    
    print(f"Found {total_files} WAV files. Processing...")
    
    total_duration = 0
    processed_files = 0
    start_time = time.time()
    durations = []  # Store individual durations for plotting
    
    for i, wav_file in enumerate(wav_files, 1):
        duration = get_wav_duration(wav_file)
        total_duration += duration
        durations.append(duration)
        processed_files += 1
        
        # Show progress every 1000 files or at the end
        if show_progress and (i % 1000 == 0 or i == total_files):
            elapsed_time = time.time() - start_time
            files_per_second = i / elapsed_time if elapsed_time > 0 else 0
            eta_seconds = (total_files - i) / files_per_second if files_per_second > 0 else 0
            
            print(f"Progress: {i}/{total_files} files ({i/total_files*100:.1f}%) - "
                  f"Speed: {files_per_second:.1f} files/sec - "
                  f"ETA: {eta_seconds/60:.1f} min")
    
    # Display results
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"{'='*50}")
    print(f"Total files processed: {processed_files}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Total duration: {hours:02d}:{minutes:02d}:{seconds:02d} (HH:MM:SS)")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"Average file duration: {total_duration/processed_files:.2f} seconds")
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    
    # Print detailed statistics
    print_duration_statistics(durations)
    
    # Create plots if requested
    if create_plots and durations:
        folder_name = folder.name if folder.name else "audio_files"
        plot_duration_distribution(durations, folder_name)

if __name__ == "__main__":
    # Your specified folder path
    folder_path = "/media/rdp/New Volume/F5-TTS/Combined_Hindi_TTS_Pure_Replacement/wavs"
    
    print(f"Processing WAV files in: {folder_path}")
    calculate_total_duration(folder_path)