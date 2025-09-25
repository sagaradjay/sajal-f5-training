#!/usr/bin/env python3
"""
Script to generate vocab.txt file from CSV data containing text column.
Extracts unique characters from the text column and saves them to vocab.txt
"""

import csv
import argparse
from pathlib import Path
from collections import Counter


def extract_unique_characters(csv_file_path, text_column='text', delimiter='|'):
    """
    Extract unique characters from text column in CSV file
    
    Args:
        csv_file_path (str): Path to the CSV file
        text_column (str): Name of the text column
        delimiter (str): CSV delimiter
    
    Returns:
        set: Set of unique characters found in the text column
    """
    unique_chars = set()
    char_counts = Counter()
    
    print(f"Reading CSV file: {csv_file_path}")
    print(f"Looking for text column: '{text_column}'")
    print(f"Using delimiter: '{delimiter}'")
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            # Try to detect delimiter if not specified
            if delimiter == 'auto':
                sample = file.read(1024)
                file.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                print(f"Auto-detected delimiter: '{delimiter}'")
            
            reader = csv.DictReader(file, delimiter=delimiter)
            
            # Check if text column exists
            if text_column not in reader.fieldnames:
                print(f"Available columns: {reader.fieldnames}")
                raise ValueError(f"Text column '{text_column}' not found in CSV file")
            
            row_count = 0
            for row in reader:
                row_count += 1
                text = row.get(text_column, '').strip()
                
                if text:
                    # Add all characters from this text to our set
                    chars_in_text = list(text)
                    unique_chars.update(chars_in_text)
                    char_counts.update(chars_in_text)
                
                # Progress indicator
                if row_count % 10000 == 0:
                    print(f"Processed {row_count} rows, found {len(unique_chars)} unique characters so far...")
            
            print(f"Finished processing {row_count} rows")
            
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise
    
    return unique_chars, char_counts


def save_vocab_file(unique_chars, output_path, sort_chars=True):
    """
    Save unique characters to vocab.txt file
    
    Args:
        unique_chars (set): Set of unique characters
        output_path (str): Path where to save vocab.txt
        sort_chars (bool): Whether to sort characters before saving
    """
    # Convert to list and optionally sort
    char_list = list(unique_chars)
    if sort_chars:
        char_list.sort()
    
    # Ensure space is first (common requirement for tokenizers)
    if ' ' in char_list:
        char_list.remove(' ')
        char_list.insert(0, ' ')
    
    print(f"Saving vocabulary to: {output_path}")
    print(f"Total unique characters: {len(char_list)}")
    
    # Show preview of characters
    print(f"Character preview (first 20): {char_list[:20]}")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            for char in char_list:
                file.write(char + '\n')
        
        print(f"Successfully saved vocab.txt with {len(char_list)} characters")
        
    except Exception as e:
        print(f"Error saving vocab file: {e}")
        raise


def print_character_statistics(char_counts, top_n=20):
    """
    Print statistics about character frequency
    
    Args:
        char_counts (Counter): Character frequency counter
        top_n (int): Number of top characters to show
    """
    print(f"\nCharacter frequency statistics:")
    print(f"Total characters processed: {sum(char_counts.values())}")
    print(f"Unique characters: {len(char_counts)}")
    
    print(f"\nTop {top_n} most frequent characters:")
    for char, count in char_counts.most_common(top_n):
        # Show printable representation
        if char == ' ':
            display_char = "' ' (space)"
        elif char == '\n':
            display_char = "'\\n' (newline)"
        elif char == '\t':
            display_char = "'\\t' (tab)"
        else:
            display_char = f"'{char}'"
        
        print(f"  {display_char}: {count} occurrences")


def main():
    parser = argparse.ArgumentParser(description='Generate vocab.txt from CSV text data')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--text-column', default='text', help='Name of the text column (default: text)')
    parser.add_argument('--delimiter', default='|', help='CSV delimiter (default: |)')
    parser.add_argument('--output', default='vocab.txt', help='Output vocab file path (default: vocab.txt)')
    parser.add_argument('--no-sort', action='store_true', help='Do not sort characters in output')
    parser.add_argument('--stats', action='store_true', help='Show character frequency statistics')
    
    args = parser.parse_args()
    
    # Validate input file
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file '{csv_path}' does not exist")
        return 1
    
    # Extract unique characters
    try:
        unique_chars, char_counts = extract_unique_characters(
            csv_path, 
            args.text_column, 
            args.delimiter
        )
        
        if not unique_chars:
            print("No characters found in the text column!")
            return 1
        
        # Save vocab file
        save_vocab_file(unique_chars, args.output, not args.no_sort)
        
        # Show statistics if requested
        if args.stats:
            print_character_statistics(char_counts)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
