import os

# Point to your kaggle config directory BEFORE importing KaggleApi
os.environ['KAGGLE_CONFIG_DIR'] = '/media/rdp/New Volume/F5-TTS/.kaggle'

from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize and authenticate
api = KaggleApi()
api.authenticate()

# Download directly to current directory
print("Starting download...")
api.dataset_download_files('skywalker290/tts-hindi-f', path='.', unzip=True)
print("Download completed in current directory")

# List the downloaded files
import glob
files = glob.glob('./**', recursive=True)
for f in files[:10]:  # Show first 10 files
    print(f) 