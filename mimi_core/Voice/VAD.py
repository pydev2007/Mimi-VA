import pyaudio
import numpy as np
import time
from scipy.io.wavfile import write

class VoiceService:
    def __init__(self): 
        self.FORMAT = pyaudio.paInt16  # Audio format (16-bit)
        self.CHANNELS = 1              # Mono audio
        self.RATE = 44100              # Sampling rate (samples per second)
        self.CHUNK = 1024              # Buffer size (number of samples per chunk)
        self.THRESHOLD = 700           # RMS threshold to consider as "quiet"
        self.QUIET_DURATION = 3        # Duration in seconds to wait before quitting if quiet
        print("Starting Service")
        self.p = pyaudio.PyAudio()

    def stop_service(self):
        self.stream.stop_stream()
        audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)
        write('test.wav', self.RATE, audio_data)
        self.stream.close()
        self.p.terminate()
    
    def start_service(self):
        self.frames = []
        self.quiet_start_time = None
        duration = 10
        start_time = time.time()

        # Open the stream
        self.stream = self.p.open(format=self.FORMAT,
                                channels=self.CHANNELS,
                                rate=self.RATE,
                                input=True,
                                frames_per_buffer=self.CHUNK)

        print("Recording...")

        while time.time() - start_time < duration:
            # Read audio data from the stream
            data = self.stream.read(self.CHUNK)

            self.frames.append(data)
            # Convert binary data to numpy array
            audio_data = np.frombuffer(data, dtype=np.int16)
            # Calculate the RMS value (a measure of loudness)
            rms = np.sqrt(np.mean(audio_data**2))
            # Print the RMS value
            print(f"RMS: {rms}")

            # Check if the audio is quiet
            if rms < self.THRESHOLD:
                if self.quiet_start_time is None:
                    # Start the timer if it's the first quiet moment
                    self.quiet_start_time = time.time()
                else:
                    # Check if the quiet duration has exceeded the limit
                    if time.time() - self.quiet_start_time > self.QUIET_DURATION:
                        print(f"Quiet for too long ({self.QUIET_DURATION} seconds). Exiting...")
                        break
            else:
                # Reset the quiet start time if the audio is loud again
                self.quiet_start_time = None

        self.stop_service()