from transformers import BertTokenizer, BertModel
import torch

# Load BERT model & tokenizer once
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def run_bert_interactively():
    print("\n🚀 BERT Interactive Mode: Type a sentence and press Enter.")
    print("🔹 Type 'exit' to quit.\n")

    while True:
        # Get user input
        text = input("Enter text: ").strip()
        
        # Exit condition
        if text.lower() == "exit":
            print("\n👋 Exiting BERT interactive mode.")
            break

        # Tokenize the input
        tokens = tokenizer(text, return_tensors="pt")

        # Run through BERT (disable gradient calculations for efficiency)
        with torch.no_grad():
            outputs = model(**tokens)

        # Extract embeddings
        embeddings = outputs.last_hidden_state

        # Print results
        print("\n📌 **Results:**")
        print("🔹 Tokens:", tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]))
        print("🔹 Embeddings Shape:", embeddings.shape, "(Batch size, Token count, 768D)\n")

# Run interactive mode
run_bert_interactively()
