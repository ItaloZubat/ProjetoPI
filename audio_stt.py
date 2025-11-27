import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json

class SpeechToText:
    def __init__(self, model_path):
        print("Carregando modelo Vosk...")
        self.model = Model(model_path)

        # Detecta dispositivo padrão do microfone
        device_info = sd.query_devices(kind="input")
        samplerate = int(device_info["default_samplerate"])

        print(f"Microfone detectado: {device_info['name']} ({samplerate} Hz)")

        self.samplerate = samplerate
        self.recognizer = KaldiRecognizer(self.model, samplerate)

        self.q = queue.Queue()

    def audio_callback(self, indata, frames, time, status):
        self.q.put(bytes(indata))

    def start(self):
        self.stream = sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=int(self.samplerate / 2),
            dtype="int16",
            channels=1,
            callback=self.audio_callback
        )
        self.stream.start()
        print("Captura de áudio iniciada!")

    def get_text(self):
        while not self.q.empty():
            data = self.q.get()
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()
                return text
        return ""
