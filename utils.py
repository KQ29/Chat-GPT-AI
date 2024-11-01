# utils.py

import config
import os

def log_session(user_input, response):
    """Logs each interaction to a session log file."""
    with open(config.SESSION_LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(f"You: {user_input}\n")
        log_file.write(f"Co-Pilot: {response}\n\n")

def handle_command_shortcuts(user_input):
    """Handles specific commands for assistance or session control."""
    user_input_lower = user_input.lower().strip()
    if user_input_lower in ["exit", "quit", "bye"]:
        return "exit"
    elif user_input_lower in ["/help", "help"]:
        return (
            "Available commands:\n"
            "/help or help - Show available commands\n"
            "/clear or clear - Clear session log\n"
            "/reset or reset - Reset conversation\n"
            "Type 'exit' to close the assistant."
        )
    elif user_input_lower in ["/clear", "clear"]:
        # Clear the session log file
        with open(config.SESSION_LOG_PATH, "w", encoding="utf-8") as log_file:
            log_file.write("")  # Clears log contents
        return "Session log cleared."
    elif user_input_lower in ["/reset", "reset"]:
        # Reset conversation history
        return "reset"
    return None

def ensure_log_directory_exists():
    """Ensures the log directory exists, creating it if necessary."""
    log_dir = os.path.dirname(config.SESSION_LOG_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
