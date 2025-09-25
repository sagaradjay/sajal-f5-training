import os
import re
import time
import schedule
import requests
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('telegram_sender.log'),
        logging.StreamHandler()
    ]
)

# Telegram configuration
TELEGRAM_BOT_TOKEN = "7013485878:AAFHsYaINpz5df3_LZTHPnVNWBzyeSg7HPE"
TELEGRAM_CHAT_ID = "-1001759470723"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Directory path
SAMPLES_DIR = "/media/rdp/New Volume/F5-TTS/ckpts/F5TTS_Hindi_Scratch_vocos_custom_IndicTTS_Hindi_CSV_char/samples"

def get_latest_checkpoint_file():
    """
    Find the latest checkpoint file matching the pattern update_<step>_gen.wav
    Returns the file path and step number, or None if no files found
    """
    try:
        samples_path = Path(SAMPLES_DIR)
        if not samples_path.exists():
            logging.error(f"Directory does not exist: {SAMPLES_DIR}")
            return None, None
        
        # Pattern to match update_<number>_gen.wav
        pattern = re.compile(r'update_(\d+)_gen\.wav')
        
        checkpoint_files = []
        for file_path in samples_path.glob("*.wav"):
            match = pattern.match(file_path.name)
            if match:
                step_number = int(match.group(1))
                checkpoint_files.append((file_path, step_number))
        
        if not checkpoint_files:
            logging.warning("No checkpoint files found matching pattern update_*_gen.wav")
            return None, None
        
        # Sort by step number and get the latest
        checkpoint_files.sort(key=lambda x: x[1], reverse=True)
        latest_file, latest_step = checkpoint_files[0]
        
        logging.info(f"Latest checkpoint found: {latest_file.name} (step {latest_step})")
        return latest_file, latest_step
        
    except Exception as e:
        logging.error(f"Error finding latest checkpoint: {e}")
        return None, None

def send_telegram_message(message):
    """Send a text message via Telegram"""
    try:
        url = f"{TELEGRAM_API_URL}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data, timeout=30)
        response.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Error sending message: {e}")
        return False

def send_telegram_audio(file_path, caption=""):
    """Send an audio file via Telegram"""
    try:
        url = f"{TELEGRAM_API_URL}/sendAudio"
        
        with open(file_path, 'rb') as audio_file:
            files = {'audio': audio_file}
            data = {
                'chat_id': TELEGRAM_CHAT_ID,
                'caption': caption,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, files=files, data=data, timeout=120)
            response.raise_for_status()
            
        logging.info(f"Successfully sent audio file: {file_path.name}")
        return True
        
    except Exception as e:
        logging.error(f"Error sending audio file: {e}")
        return False

def send_latest_checkpoint():
    """Find and send the latest checkpoint file"""
    logging.info("Checking for latest checkpoint...")
    
    latest_file, latest_step = get_latest_checkpoint_file()
    
    if latest_file is None:
        error_msg = "❌ No checkpoint files found!"
        logging.warning(error_msg)
        send_telegram_message(error_msg)
        return
    
    # Check file size
    file_size_mb = latest_file.stat().st_size / (1024 * 1024)
    
    # Get file modification time
    mod_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
    
    caption = f"""🎵 <b>Training Checkpoint Update</b>
    
📁 <b>File:</b> {latest_file.name}
🔢 <b>Step:</b> {latest_step:,}
📊 <b>Size:</b> {file_size_mb:.2f} MB
🕐 <b>Created:</b> {mod_time.strftime('%Y-%m-%d %H:%M:%S')}
⏰ <b>Sent:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
    
    # Send the audio file
    if send_telegram_audio(latest_file, caption):
        logging.info(f"Successfully sent checkpoint step {latest_step}")
    else:
        error_msg = f"❌ Failed to send checkpoint step {latest_step}"
        logging.error(error_msg)
        send_telegram_message(error_msg)

def test_telegram_connection():
    """Test if Telegram bot is working"""
    try:
        url = f"{TELEGRAM_API_URL}/getMe"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        bot_info = response.json()
        
        if bot_info.get('ok'):
            bot_name = bot_info['result'].get('username', 'Unknown')
            logging.info(f"✅ Telegram bot connected successfully: @{bot_name}")
            send_telegram_message("🤖 <b>Checkpoint Monitor Started</b>\n\nThe training checkpoint monitor is now active and will send updates every 30 minutes.")
            return True
        else:
            logging.error("❌ Telegram bot connection failed")
            return False
            
    except Exception as e:
        logging.error(f"❌ Error testing Telegram connection: {e}")
        return False

def main():
    """Main function to set up and run the scheduler"""
    print("🚀 Starting Telegram Training Checkpoint Monitor...")
    
    # Test Telegram connection
    if not test_telegram_connection():
        print("❌ Failed to connect to Telegram. Please check your bot token and chat ID.")
        return
    
    # Test directory access
    if not Path(SAMPLES_DIR).exists():
        print(f"❌ Directory not found: {SAMPLES_DIR}")
        return
    
    # Schedule the job to run every 30 minutes
    schedule.every(30).minutes.do(send_latest_checkpoint)
    
    # Send initial checkpoint immediately
    print("📤 Sending initial checkpoint...")
    send_latest_checkpoint()
    
    print("⏰ Scheduler started. Will send updates every 30 minutes.")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping checkpoint monitor...")
        send_telegram_message("🛑 <b>Checkpoint Monitor Stopped</b>\n\nThe training checkpoint monitor has been stopped.")
        logging.info("Checkpoint monitor stopped by user")

if __name__ == "__main__":
    main()