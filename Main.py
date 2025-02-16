from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.core.window import Window
import speech_recognition as sr
from utils.RTAS import record_transcribe_save

class VoiceNoteApp(App):
    def build(self):
        self.recognizer = sr.Recognizer()
        layout = BoxLayout(orientation='vertical')
        self.label = Label(text="Recording...")
        layout.add_widget(self.label)
        Clock.schedule_once(self.start_recording, 1)  # Start recording after 1 second
        Window.bind(on_touch_up=self.on_touch_up)  # Bind touch event to stop recording
        return layout

    def start_recording(self, dt):
        self.label.text = "Listening..."
        text = record_transcribe_save(self.recognizer, timeout=5)  # Record, transcribe, and save
        self.label.text = "You said: " + text
        Clock.schedule_once(self.stop_recording, 1)  # Schedule stop recording after processing

    def on_touch_up(self, *args):
        self.stop_recording()

    def stop_recording(self, *args):
        self.label.text = "Recording stopped."
        App.get_running_app().stop()  # Close the application

if __name__ == "__main__":
    VoiceNoteApp().run()