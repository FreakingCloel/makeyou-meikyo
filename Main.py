import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
import speech_recognition as sr

class VoiceNoteApp(App):
    def build(self):
        self.recognizer = sr.Recognizer()
        layout = BoxLayout(orientation='vertical')
        self.label = Label(text="Press the button and start speaking")
        self.button = Button(text="Record", on_press=self.record_voice_note)
        layout.add_widget(self.label)
        layout.add_widget(self.button)
        return layout

    def record_voice_note(self, instance):
        with sr.Microphone() as source:
            self.label.text = "Listening..."
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio)
                self.label.text = "You said: " + text
            except sr.UnknownValueError:
                self.label.text = "Sorry, I could not understand the audio."
            except sr.RequestError:
                self.label.text = "Could not request results; check your network connection."

if __name__ == "__main__":
    VoiceNoteApp().run()