from pyt2s.services import stream_elements
import pathlib
# import pyttsx3


class text_to_speech:
    def __init__(self):
         
        pass

    def fallback_tts(self):
        # engine = pyttsx3.init()
        pass

    def SE_tts(self, text):
        # Custom Voice
        data = stream_elements.requestTTS(text, stream_elements.Voice.Brian.value)

        with open(f"{pathlib.Path(__file__).parent.resolve()}\\output.mp3", '+wb') as file:
            file.write(data)

