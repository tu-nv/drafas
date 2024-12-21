import random
from PIL import Image
import numpy as np
import wave
import io
import torchvision
import torchvision.transforms as transforms
import torchaudio
from datasets import load_dataset

random.seed(42)

# preprocessing image to fit input of resnet50 in triton
def preprocess_img(img: Image.Image, dtype, h=224, w=224):
    sample_img = img.convert('RGB')
    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    resized = resized.astype(dtype)

    return resized

def create_random_audio():
    # Parameters
    sample_rate = 16000  # 16 kHz
    duration = 1  # 1 second
    num_samples = sample_rate * duration  # Total number of samples
    amplitude = 32767  # Maximum value for 16-bit PCM (int16)

    # Generate random audio data (values between -32768 and 32767)
    random_data = np.random.randint(-amplitude, amplitude, num_samples, dtype=np.int16)

    # Store the WAV data in a BytesIO object (in-memory storage)
    wav_data = io.BytesIO()

    # Create the WAV binary data
    with wave.open(wav_data, 'wb') as wav_file:
        # Set parameters for the WAV file
        wav_file.setnchannels(1)  # Mono channel
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit PCM)
        wav_file.setframerate(sample_rate)  # 16 kHz sample rate

        # Write the raw binary data to the in-memory buffer
        wav_file.writeframes(random_data.tobytes())

    # Get the binary data from BytesIO
    return wav_data.getvalue()

def load_google_speech_command_dataset():
    # Load the Google Speech Commands dataset
    dataset = load_dataset("google/speech_commands", 'v0.01')
    test_data = dataset["test"]
    audio_samples = []
    for sample in test_data:
        audio_samples.append(sample["audio"]["array"].tobytes())
    return audio_samples[:1000]

if __name__ == "__main__":
    create_random_audio()
    load_google_speech_command_dataset()




