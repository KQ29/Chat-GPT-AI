# app.py

from model_loader import load_model
from response_generator import generate_response
from utils import log_session, handle_command_shortcuts, ensure_log_directory_exists
import warnings
import os

# Подавление предупреждений
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    print("Co-Pilot: Привет! Я ваш локальный ассистент по программированию. Введите '/help' для списка команд или 'exit' для выхода.")
    ensure_log_directory_exists()
    model, tokenizer, device = load_model()
    chat_history_ids = None

    while True:
        user_input = input("Вы: ")

        # Обработка команд
        command_response = handle_command_shortcuts(user_input)
        if command_response:
            if command_response == "exit":
                print("Co-Pilot: До свидания! Удачного кодинга!")
                break
            elif command_response == "reset":
                chat_history_ids = None
                print("Co-Pilot: История общения сброшена.")
                continue
            else:
                print("Co-Pilot:", command_response)
                continue

        # Генерация и вывод ответа
        response_text, chat_history_ids = generate_response(
            model, tokenizer, device, user_input, chat_history_ids
        )
        print("Co-Pilot:", response_text)

        # Логирование сессии
        log_session(user_input, response_text)

if __name__ == "__main__":
    main()
