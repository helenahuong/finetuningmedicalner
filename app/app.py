import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

ner_labels = {
    '0': 'O',     # Outside of a named entity
    '1': 'B-DIS', # Beginning of a disease
    '2': 'I-DIS', # Inside of a disease
    '3': 'B-MED', # Beginning of a medication
    '4': 'I-MED'  # Inside of a medication
}

model = AutoModelForTokenClassification.from_pretrained("./results")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def predict(text):
    tokens = text.split()
    inputs = tokenizer(tokens, return_tensors="pt", is_split_into_words=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=2).squeeze().tolist()
    entities = [(token, ner_labels[str(pred)]) for token, pred in zip(tokens, predictions)]
    return entities

gr.Interface(fn=predict, inputs="text", outputs="text").launch()
