import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import pygame
import tempfile
import os

class VoiceProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
    def listen_to_speech(self):
        with self.microphone as source:
            print("Listening... Please describe your symptoms.")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, timeout=10)
        
        try:
            text = self.recognizer.recognize_google(audio)
            return text.lower()
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Speech recognition service unavailable."
    
    def speak_response(self, text):
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            pygame.mixer.init()
            pygame.mixer.music.load(tmp_file.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
        os.unlink(tmp_file.name)