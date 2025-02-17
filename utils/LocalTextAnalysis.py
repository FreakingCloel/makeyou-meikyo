import torch
from transformers import pipeline, BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load BERT emotion classifier
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", top_k=5)

# Load BERT model & tokenizer for embeddings
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Function to get sentence embedding
def get_sentence_embedding(text):
    tokens = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Energy-related concept embeddings (dynamic inference)
energy_concepts = {
    "movement": "I feel active and ready to move.",
    "rest": "I feel like I need to lie down and relax.",
    "fatigue": "I feel physically drained and tired.",
    "engagement": "I am focused and mentally present.",
    "motivation": "I feel inspired and ready to take on challenges.",
    "stress": "I feel overwhelmed and unable to concentrate."
}

# Precompute concept embeddings
concept_embeddings = {
    key: get_sentence_embedding(value) for key, value in energy_concepts.items()
}

# Function to infer energy levels dynamically
def extract_energy(text):
    memo_embedding = get_sentence_embedding(text)
    
    energy_scores = {
        category: cosine_similarity(memo_embedding.reshape(1, -1), ref_embedding.reshape(1, -1))[0][0]
        for category, ref_embedding in concept_embeddings.items()
    }
    
    # Normalize energy based on relevant categories
    physical_energy = 2 * (energy_scores["movement"] - energy_scores["fatigue"]) - 1
    mental_energy = 2 * (energy_scores["engagement"] - energy_scores["stress"]) - 1
    
    return {
        "physical_energy": round(physical_energy, 3),
        "mental_energy": round(mental_energy, 3)
    }

# Function to get emotion classification from BERT
def extract_sentiment(text):
    results = emotion_classifier(text)
    formatted_results = {entry['label']: entry['score'] for entry in results[0]}
    return formatted_results

# Main function to extract structured analysis
def extract_structured_keywords(text):
    emotions = extract_sentiment(text)
    energy_levels = extract_energy(text)
    
    return {
        "emotions": emotions,
        "energy_levels": energy_levels
    }

# Interactive Loop for Testing
def interactive_testing():
    print("\nType a sentence to analyze. Type 'exit' to quit.")
    while True:
        text = input("Enter text: ").strip()
        if text.lower() == "exit":
            print("Exiting interactive mode.")
            break
        analysis = extract_structured_keywords(text)
        print("\nAnalysis:", analysis, "\n")

# Run interactive mode if script is executed directly
if __name__ == "__main__":
    interactive_testing()
