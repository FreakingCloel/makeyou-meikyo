## Record, transcribe, analyze, and save

import speech_recognition as sr
import json
from datetime import datetime
import os
from .LocalTextAnalysis import extract_structured_keywords

def record_transcribe_save(recognizer, timeout=5):
    with sr.Microphone() as source:
        audio = recognizer.listen(source, timeout=timeout)
        try:
            text = recognizer.recognize_google(audio)
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            audio_filename = save_audio(audio, timestamp)
            transcript_filename = save_transcript(text, timestamp)
            analysis_filename = save_analysis(text, timestamp, audio_filename, transcript_filename)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Could not request results; check your network connection."

def save_audio(audio, timestamp):
    # Create a directory for recordings if it doesn't exist
    if not os.path.exists('Libs/Recordings'):
        os.makedirs('Libs/Recordings')

    # Generate a filename based on the current date and time
    filename = f'Libs/Recordings/{timestamp}.wav'

    # Save the audio file
    with open(filename, 'wb') as f:
        f.write(audio.get_wav_data())
    
    return filename

def save_transcript(text, timestamp):
    # Create a directory for transcripts if it doesn't exist
    if not os.path.exists('Libs/Transcripts'):
        os.makedirs('Libs/Transcripts')

    # Generate a filename based on the current date and time
    filename = f'Libs/Transcripts/{timestamp}.json'

    # Save the transcript with metadata
    data = {
        'timestamp': timestamp,
        'text': text
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    return filename

def save_analysis(text, timestamp, audio_filename, transcript_filename):
    # Create a directory for analysis references if it doesn't exist
    if not os.path.exists('Libs/Refs'):
        os.makedirs('Libs/Refs')

    # Extract structured keywords and named entities
    keywords, ner_results = extract_structured_keywords([text])

    # Generate a filename based on the current date and time
    filename = f'Libs/Refs/{timestamp}.json'

    # Save the analysis data with references to the audio and transcript files
    data = {
        'timestamp': timestamp,
        'audio_file': audio_filename,
        'transcript_file': transcript_filename,
        'keywords': keywords,
        'named_entities': ner_results
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    return filename

def generate_tags(text):
    # Placeholder for tag generation logic
    # You can implement custom logic to generate tags based on the content of the transcript
    return ['example_tag']