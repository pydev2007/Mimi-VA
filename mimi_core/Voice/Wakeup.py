import struct
import pyaudio
import pvporcupine

# TMP Imports
import VAD
import STT
import TTS
import pathlib
import vlc


import openwakeword
from openwakeword.model import Model
import sounddevice as sd
import numpy as np
import time
import struct
import pyaudio

class WakeWord:
    def __init__(self):
        self.service = VAD.VoiceService()
        config = 1
        self.listening = True

        if config == 1:
            self.OWakeWord()

    def Callback(self):
        print("done")
        pass
    def PicoVoice(self):
        # while True:
            porcupine = pvporcupine.create(access_key="", keyword_paths=[f"{pathlib.Path(__file__).parent.resolve()}\\heyMimi.ppn"])

            pa = pyaudio.PyAudio()

            audio_stream = pa.open(
                            rate=porcupine.sample_rate,
                            channels=1,
                            format=pyaudio.paInt16,
                            input=True,
                            frames_per_buffer=porcupine.frame_length)

            while True:
                pcm = audio_stream.read(porcupine.frame_length)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                keyword_index = porcupine.process(pcm)

                if keyword_index >= 0:
                    print("Hotword Detected")
                    self.service.start_service()
                    TTS.SE_tts(STT.words())
                    script_dir = pathlib.Path(__file__).parent.resolve()

                    # Construct the full path to the audio file
                    audio_path = script_dir / "output.mp3"

                    # Print the path for debugging
                    print(f"Playing sound from: {audio_path}")

                    # Check if the file exists
                    if not audio_path.exists():
                        print(f"Error: File not found at {audio_path}")
                    else:
                        try:
                            # Create a VLC instance and media player
                            instance = vlc.Instance()
                            player = instance.media_player_new()

                            # Load the audio file
                            media = instance.media_new(str(audio_path))
                            player.set_media(media)

                            # Play the audio
                            player.play()

                            # Wait for the audio to finish playing
                            print("Playback finished.")
                            break
                        except Exception as e:
                            print(f"An error occurred: {e}")
    def OWakeWord(self):
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
            while self.listening == True:
                print("Listening...")
                try:
                    # Read audio data and convert to numpy array
                    pcm = audio_stream.read(chunk_size, exception_on_overflow=False)
                    audio_data = np.frombuffer(pcm, dtype=np.int16)  # Convert bytes to numpy array
                    
                    # Get prediction
                    prediction = ow_model.predict(audio_data)
                    
                    # Check if any wake word was detected
                    if any(score > 0.5 for score in prediction.values()):  # Adjust threshold as needed
                        # print("Hotword Detected")
                        self.listening = False
                        
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

        self.Callback()

WakeWord()