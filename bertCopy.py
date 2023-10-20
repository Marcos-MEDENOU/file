from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Chargez le modèle BERT fine-tuné pour le chat
model_name = "save/saved_weights.pt"  # Remplacez par le nom du modèle que vous utilisez
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fonction pour obtenir une réponse du modèle
def get_response(question):
    # Tokenisez la question
    input_ids = tokenizer.encode(question, return_tensors="pt")

    # Générez une réponse du modèle
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

    # Décodage de la réponse
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Boucle de chat
while True:
    user_input = input("Vous: ")
    if user_input.lower() == "exit":
        break
    response = get_response(user_input)
    print("Chatbot:", response)
