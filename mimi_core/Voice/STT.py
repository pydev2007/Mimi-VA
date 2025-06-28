import wave
import json
from vosk import Model, KaldiRecognizer, SetLogLevel

def words():
    SetLogLevel(0)

    wf = wave.open("test.wav", "rb")

    model = Model(model_name="vosk-model-en-us-0.22-lgraph")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

                    
    text = []    
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        # if silence detected save result
        if rec.AcceptWaveform(data):
            text.append(json.loads(rec.Result())["text"])
    text.append(json.loads(rec.FinalResult())["text"])
    print(text[0])

    return text[0]
