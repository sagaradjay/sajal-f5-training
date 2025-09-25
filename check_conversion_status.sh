#!/bin/bash

INPUT_DIR="/media/rdp/New Volume/F5-TTS/Combined_Hindi_TTS_Raw_Data/wavs"
OUTPUT_DIR="/media/rdp/New Volume/F5-TTS/Combined_Hindi_TTS_Raw_Data/wavs_converted_24k"

echo "=== Conversion Status Report ==="
echo ""

total_input=$(find "$INPUT_DIR" -name "*.wav" | wc -l)
total_output=$(find "$OUTPUT_DIR" -name "*.wav" | wc -l)

echo "Total input files: $total_input"
echo "Total converted files: $total_output"
echo "Remaining files: $((total_input - total_output))"
echo "Progress: $(( (total_output * 100) / total_input ))%"
echo ""

if [ $total_output -gt 0 ]; then
    echo "=== Sample of converted files ==="
    find "$OUTPUT_DIR" -name "*.wav" | head -5 | while read -r file; do
        filename=$(basename "$file")
        size=$(stat -c%s "$file")
        echo "  $filename: $size bytes"
        
        # Check audio properties
        sample_rate=$(ffprobe -v quiet -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 "$file")
        channels=$(ffprobe -v quiet -select_streams a:0 -show_entries stream=channels -of csv=p=0 "$file")
        echo "    Sample rate: ${sample_rate}Hz, Channels: $channels"
    done
fi
