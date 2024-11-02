# response_generator.py

import torch
import re
from sympy import sympify, SympifyError
from sympy import sin, cos, tan, pi, E, exp, sqrt, log

def is_math_expression(text):
    """Determines if the input text is a mathematical expression."""
    try:
        expression = preprocess_expression(text)
        expr = sympify(expression)
        # If the expression has free symbols (undefined variables), it's likely not purely mathematical
        if expr.free_symbols:
            return False
        else:
            return True
    except SympifyError:
        return False
    except Exception:
        return False

def preprocess_expression(expression):
    """Preprocesses the mathematical expression for evaluation."""
    expression = expression.lower()
    # Replace English words with mathematical operators
    expression = expression.replace('^', '**')  # Exponentiation
    expression = expression.replace('of', '*')  # For phrases like '10% of 200'
    expression = expression.replace('%', '/100')  # Percentages
    # Remove spaces
    expression = expression.replace(' ', '')
    return expression

def evaluate_math_expression(expression):
    """Evaluates the mathematical expression safely using sympy."""
    try:
        expression = preprocess_expression(expression)
        expr = sympify(expression)
        if expr.free_symbols:
            return "I'm sorry, I couldn't evaluate that expression."
        result = expr.evalf()
        return str(result)
    except SympifyError:
        return "I'm sorry, I couldn't evaluate that expression."
    except Exception:
        return "An error occurred while evaluating the expression."

def generate_response(model, tokenizer, device, user_input, chat_history_ids=None):
    """Generates a response from the model based on user input."""
    # Ensure pad_token_id is defined and different from eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    stripped_input = user_input.rstrip('?.! ')

    # Check if the user input is a math expression
    if is_math_expression(stripped_input):
        response_text = evaluate_math_expression(stripped_input)
        return response_text, chat_history_ids

    # Extract expressions from phrases like "what is 4 + 4"
    match = re.search(r'(?:what|how much|calculate)\s+(?:is|will be)\s+(.+)', user_input.lower())
    if match:
        expression = match.group(1)
        if is_math_expression(expression):
            response_text = evaluate_math_expression(expression)
            return response_text, chat_history_ids

    # Proceed with generating a response using the model
    new_user_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token, return_tensors='pt'
    ).to(device)

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.75,
    )

    response_text = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True
    )
    return response_text, chat_history_ids
