# model_loader.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import config
import torch

def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model loaded successfully!")
    return model, tokenizer, device
