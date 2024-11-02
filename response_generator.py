# response_generator.py

import re
from sympy import sympify, SympifyError
from model_loader import query_huggingface_api
import config

def is_math_expression(text):
    """
    Determines if the input text is a mathematical expression.
    Returns True if it is, False otherwise.
    """
    math_keywords = ['calculate', 'compute', 'evaluate', 'solve', 'what is', 'what\'s', 'find']

    # Exclude inputs ending with a question mark to prevent misclassification
    if text.strip().endswith('?'):
        return False

    # Check for math-related keywords
    if any(keyword in text.lower() for keyword in math_keywords):
        return True

    # Attempt to parse the text as a mathematical expression
    try:
        expression = preprocess_expression(text)
        expr = sympify(expression)

        # If the expression has free symbols, it's likely not purely mathematical
        if expr.free_symbols:
            return False
        else:
            return True
    except SympifyError:
        return False
    except Exception:
        return False

def preprocess_expression(expression):
    """
    Preprocesses the mathematical expression for evaluation.
    """
    expression = expression.lower()
    # Replace common words/operators to standard mathematical symbols
    expression = expression.replace('^', '**')        # Exponentiation
    expression = expression.replace('of', '*')        # For phrases like '10% of 200'
    expression = expression.replace('%', '/100')      # Percentages
    # Remove spaces and other non-essential characters
    expression = re.sub(r'\s+', '', expression)
    return expression

def evaluate_math_expression(expression):
    """
    Evaluates the mathematical expression safely using sympy.
    Returns the result as a string.
    """
    try:
        expression = preprocess_expression(expression)
        expr = sympify(expression)

        if expr.free_symbols:
            return "I'm sorry, I couldn't evaluate that expression because it contains variables."

        result = expr.evalf()

        # Format the result to remove unnecessary decimal points
        if result == int(result):
            result = int(result)
        else:
            result = float(result)

        return str(result)
    except SympifyError:
        return "I'm sorry, I couldn't evaluate that expression."
    except Exception:
        return "An error occurred while evaluating the expression."

def handle_math_expression(user_input, chat_history=None):
    """
    Handles mathematical expressions by evaluating them.
    Returns the result and updated chat history.
    """
    result = evaluate_math_expression(user_input)
    return result, chat_history

def generate_response(user_input, chat_history=None, use_api=True):
    """
    Generates a response using the Hugging Face Inference API if available.
    If the API isn't accessible, switches to calculator functionality.
    Returns the response text and updated chat history.
    """
    stripped_input = user_input.strip()

    if use_api:
        # Use the Hugging Face Inference API to generate a response
        payload = {
            "inputs": stripped_input,
            "parameters": {
                "max_length": config.MAX_LENGTH,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.95,
                "no_repeat_ngram_size": 3
            }
        }
        response_text = query_huggingface_api(payload)
        return response_text, chat_history
    else:
        # Calculator functionality
        if is_math_expression(stripped_input):
            result = evaluate_math_expression(stripped_input)
            return result, chat_history
        else:
            return "I'm sorry, I can only help with mathematical calculations. Please enter a math problem.", chat_history
