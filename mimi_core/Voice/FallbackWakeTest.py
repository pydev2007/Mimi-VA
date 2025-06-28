print("start")
import openwakeword
from openwakeword.model import Model
import sounddevice as sd
import numpy as np
import time
import struct
import pyaudio
print("modules imported")

openwakeword.utils.download_models()

# Initialize the model with your custom wake word
ow_model = Model(
    wakeword_models=["D:\Programming\Mimi\mimi_core\Voice\hey_mimi.onnx"],
)

# Set up audio parameters
samplerate = 16000
chunk_size = 400  # Should match your model's expected input size

print("got model")

# Initialize PyAudio outside the loop
pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=samplerate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=chunk_size
)

try:
    while True:
        print("Listening...")
        try:
            # Read audio data and convert to numpy array
            pcm = audio_stream.read(chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(pcm, dtype=np.int16)  # Convert bytes to numpy array
            
            # Get prediction
            prediction = ow_model.predict(audio_data)
            
            # Check if any wake word was detected
            if any(score > 0.5 for score in prediction.values()):  # Adjust threshold as needed
                print("Hotword Detected")
                
        except Exception as e:
            print(f"Error in audio processing: {e}")
            time.sleep(0.1)  # Brief pause if error occurs
            
except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Clean up resources
    audio_stream.stop_stream()
    audio_stream.close()
    pa.terminate()