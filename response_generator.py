import torch
import re
from sympy import sympify

def is_math_expression(text):
    text = text.rstrip('?.! ')
    pattern = r'^\s*[\d\s+\-*/%.\(\)]+(?:\s*(?:percent|of)\s*[\d\s+\-*/%.\(\)]+)*\s*$'
    return bool(re.match(pattern, text.lower()))

def preprocess_expression(expression):
    expression = expression.lower()
    expression = expression.replace('percent', '/100')
    expression = expression.replace('of', '*')
    expression = re.sub(r'(\d+)\s*%', r'(\1/100)', expression)
    expression = expression.replace(' ', '')
    return expression

def evaluate_math_expression(expression):
    try:
        expression = preprocess_expression(expression)
        result = sympify(expression).evalf()
        return str(result)
    except Exception:
        return "I'm sorry, I couldn't evaluate that expression."

def generate_response(model, tokenizer, device, user_input, chat_history_ids=None):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    stripped_input = user_input.rstrip('?.! ')

    if is_math_expression(stripped_input):
        response_text = evaluate_math_expression(stripped_input)
        return response_text, chat_history_ids

    match = re.search(r'what\s+(?:is|will)\s+([\d\s+\-*/%.\(\)]+)', user_input.lower())
    if match:
        expression = match.group(1)
        if is_math_expression(expression):
            response_text = evaluate_math_expression(expression)
            return response_text, chat_history_ids

    # Encode user input and generate response as before
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
