import torch
from transformers import pipeline, BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load BERT emotion classifier
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotions", top_k=3)

# Load BERT model & tokenizer for embeddings
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Function to get emotion classification from BERT
def extract_sentiment(text):
    results = emotion_classifier(text)
    return {entry['label']: entry['score'] for entry in results[0]}

# Function to get sentence embedding for energy analysis
def get_sentence_embedding(text):
    tokens = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Function to estimate energy levels
def extract_energy(text):
    energy_vectors = {
        "high_energy": np.random.rand(768),  # Placeholder vector
        "low_energy": np.random.rand(768)   # Placeholder vector
    }
    memo_embedding = get_sentence_embedding(text)
    energy_scores = {level: cosine_similarity(memo_embedding.reshape(1, -1), vector.reshape(1, -1))[0][0] for level, vector in energy_vectors.items()}
    return max(energy_scores, key=energy_scores.get)

# Main function to extract structured keywords (now includes sentiment & energy analysis)
def extract_structured_keywords(text):
    emotions = extract_sentiment(text)
    energy_level = extract_energy(text)
    
    return {
        "emotions": emotions,
        "energy_level": energy_level
    }

# Example usage
if __name__ == "__main__":
    text = "I'm feeling super tired today but also kind of excited about my project."
    analysis = extract_structured_keywords(text)
    print("Analysis:", analysis)
