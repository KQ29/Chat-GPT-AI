# model_loader.py

import config
import requests
import time

def query_huggingface_api(payload, retries=3, backoff_factor=2):
    """
    Sends a POST request to the Hugging Face Inference API with the given payload.
    Implements retry logic with exponential backoff.
    Returns the response text if successful, else returns an error message.
    """
    headers = {
        "Authorization": f"Bearer {config.HUGGINGFACE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    for attempt in range(retries):
        try:
            response = requests.post(config.HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=60)  # Increased timeout for larger models
            response.raise_for_status()
            # Extract 'generated_text' from the response
            data = response.json()
            if isinstance(data, list) and 'generated_text' in data[0]:
                return data[0]['generated_text']
            elif isinstance(data, dict) and 'generated_text' in data:
                return data['generated_text']
            else:
                return "I'm sorry, I couldn't process your request at the moment."
        except requests.exceptions.HTTPError as http_err:
            if response.status_code != 429:
                break
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.Timeout:
            pass
        except Exception:
            pass

        # Retry after backoff if not the last attempt
        if attempt < retries - 1:
            sleep_time = backoff_factor ** attempt
            time.sleep(sleep_time)
        else:
            return "I'm sorry, I couldn't process your request at the moment."

    return "I'm sorry, I couldn't process your request at the moment."
