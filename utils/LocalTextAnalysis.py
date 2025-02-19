import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# Load the emotion classification model and tokenizer
MODEL_NAME = "bhadresh-savani/bert-base-go-emotion"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Fatigue Likelihood Fixed Scores
emotion_heavy_fatigue = 80
emotion_medium_fatigue = 60
emotion_neutral_fatigue = 40
emotion_low_fatigue = 20

# Fatigue Likelihood Matrix
fatigue_likelihood = {
    "admiration": emotion_low_fatigue, "amusement": emotion_low_fatigue, "anger": emotion_heavy_fatigue, "annoyance": emotion_medium_fatigue, "approval": emotion_neutral_fatigue, "caring": emotion_neutral_fatigue, "confusion": emotion_heavy_fatigue, "curiosity": emotion_low_fatigue, "desire": 25,
    "disappointment": emotion_medium_fatigue, "disapproval": emotion_heavy_fatigue, "disgust": emotion_heavy_fatigue, "embarrassment": emotion_heavy_fatigue, "excitement": emotion_neutral_fatigue, "fear": emotion_heavy_fatigue, "gratitude": emotion_neutral_fatigue, "grief": emotion_heavy_fatigue,
    "joy": emotion_low_fatigue, "love": emotion_low_fatigue, "nervousness": emotion_heavy_fatigue, "neutral": emotion_neutral_fatigue, "optimism": emotion_neutral_fatigue, "pride": emotion_neutral_fatigue, "realization": emotion_low_fatigue
    , "relief": emotion_neutral_fatigue, "remorse": emotion_heavy_fatigue, "sadness": emotion_medium_fatigue, "surprise": emotion_medium_fatigue
}


# Memo Energy Level Matrix (Trinary)
memo_energy_levels = {
    "admiration": "Medium", "amusement": "High", "anger": "High", "annoyance": "Medium", "approval": "Medium", "caring": "Medium", "confusion": "Medium", "curiosity": "High", "desire": "High",
    "disappointment": "Low", "disapproval": "Medium", "disgust": "Medium", "embarrassment": "Medium", "excitement": "High", "fear": "High", "gratitude": "Medium", "grief": "Low", "joy": "High", "love": "Medium",
    "nervousness": "High", "neutral": "Medium", "optimism": "High", "pride": "High", "realization": "Medium", "relief": "Medium", "remorse": "Low", "sadness": "Low", "surprise": "High"
}

# Function to analyze a memo for fatigue likelihood and memo energy classification
def analyze_memo_energy(emotion_scores: dict):
    """
    Given emotion scores (output from BERT), compute:
    1. Fatigue likelihood (percentage)
    2. Memo energy classification (High, Medium, Low)
    """
    # Select the top 5 emotions by confidence
    top_5_emotions = dict(sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:5])
    
    # Compute fatigue likelihood by averaging fatigue values for top 5 emotions
    fatigue_values = [fatigue_likelihood.get(emotion, 0) for emotion in top_5_emotions]
    fatigue_score = sum(fatigue_values) / len(fatigue_values) if fatigue_values else 0

    print(f"Fatigue Values Used: {fatigue_values}")  # Add this inside analyze_memo_energy()
    
    # Compute memo energy classification based on top 5 emotions
    energy_counts = {"High": 0, "Medium": 0, "Low": 0}
    for emotion in top_5_emotions:
        if emotion in memo_energy_levels:
            energy_counts[memo_energy_levels[emotion]] += 1
    memo_energy = max(energy_counts, key=energy_counts.get)
    
    return fatigue_score, memo_energy, top_5_emotions

# Function to classify emotions using BERT
def classify_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
    labels = model.config.id2label
    return {labels[i]: scores[i] for i in range(len(scores))}

# Terminal loop for user input
if __name__ == "__main__":
    print("Emotion Analysis Terminal - Type a sentence and press Enter (Type 'exit' to quit)")
    while True:
        user_input = input("Enter your memo: ")
        if user_input.lower() == 'exit':
            break
        emotion_scores = classify_emotions(user_input)
        fatigue, memo_energy, top_5_emotions = analyze_memo_energy(emotion_scores)
        
        print(f"\nTop 5 Detected Emotions:")
        for emotion, score in top_5_emotions.items():
            print(f"  {emotion}: {score:.2f}")
        
        print(f"\nFatigue Score: {fatigue}%")
        print(f"Memo Energy Level: {memo_energy}")
        print("\n----------------------------------------------------")
