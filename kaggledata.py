import kagglehub

# Download latest version
path = kagglehub.dataset_download("skywalker290/tts-hindi-f")

print("Path to dataset files:", path)