# response_generator.py

import torch
import re
import numexpr as ne

def is_math_expression(text):
    text = text.rstrip('?.! ')
    # Расширенный паттерн для распознавания математических выражений
    pattern = r'^[\d\s\+\-\*/\%\.\(\)eE]+$'
    return bool(re.match(pattern, text))

def preprocess_expression(expression):
    expression = expression.lower()
    expression = expression.replace('^', '**')  # Замена возведения в степень
    expression = expression.replace(' ', '')
    return expression

def evaluate_math_expression(expression):
    try:
        expression = preprocess_expression(expression)
        # Безопасное вычисление математического выражения с помощью numexpr
        result = ne.evaluate(expression)
        return str(result)
    except Exception:
        return "Извините, не удалось вычислить выражение."

def generate_response(model, tokenizer, device, user_input, chat_history_ids=None):
    # Установка pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    stripped_input = user_input.rstrip('?.! ')

    # Проверка на математическое выражение
    if is_math_expression(stripped_input):
        response_text = evaluate_math_expression(stripped_input)
        return response_text, chat_history_ids

    # Обработка фраз типа "сколько будет 4 + 4"
    match = re.search(r'(?:сколько|что|чему)\s+(?:будет|равно)\s+(.+)', user_input.lower())
    if match:
        expression = match.group(1)
        if is_math_expression(expression):
            response_text = evaluate_math_expression(expression)
            return response_text, chat_history_ids

    # Генерация ответа с помощью модели
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
