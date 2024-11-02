# app.py

from response_generator import generate_response, is_math_expression, handle_math_expression
from utils import handle_command_shortcuts, ensure_log_directory_exists
import config
import requests
import os
import warnings
import time

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    print("Co-Pilot: Hi! I'm your local chat assistant. Type '/help' for commands or 'exit' to quit.")
    ensure_log_directory_exists()
    chat_history = []
    api_available = False
    last_check = 0
    check_interval = 60  # seconds

    while True:
        user_input = input("You: ").strip()

        # Handle command shortcuts
        command_response = handle_command_shortcuts(user_input)
        if command_response:
            if command_response == "exit":
                print("Co-Pilot: Goodbye! Have a great day!")
                break
            elif command_response == "reset":
                chat_history = []
                print("Co-Pilot: Conversation history has been reset.")
                continue
            else:
                print("Co-Pilot:", command_response)
                continue

        # Determine if the input is a mathematical expression
        if is_math_expression(user_input):
            response_text, chat_history = handle_math_expression(user_input, chat_history)
            print("Co-Pilot:", response_text)
            continue

        # Check API availability periodically
        current_time = time.time()
        if current_time - last_check > check_interval:
            try:
                test_payload = {
                    "inputs": "Hello, how are you?",
                    "parameters": {
                        "max_length": config.MAX_LENGTH,
                        "temperature": 0.7,
                        "top_k": 50,
                        "top_p": 0.95,
                        "no_repeat_ngram_size": 3
                    }
                }
                test_response = requests.post(
                    config.HUGGINGFACE_API_URL,
                    headers={"Authorization": f"Bearer {config.HUGGINGFACE_API_TOKEN}"},
                    json=test_payload,
                    timeout=10  # seconds
                )
                if test_response.status_code == 200:
                    api_available = True
                else:
                    api_available = False
            except Exception:
                api_available = False
            last_check = current_time

        # Generate and display response using API if available
        response_text, chat_history = generate_response(
            user_input, chat_history, use_api=api_available
        )
        print("Co-Pilot:", response_text)

if __name__ == "__main__":
    main()
