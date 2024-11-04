import time

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted


def make_request(
    client: genai.GenerativeModel,
    messages: List,
    temperature: float,
    n: int,
    max_new_tokens: int = 2048,
) -> genai.types.GenerateContentResponse:
    messages = [{"role": m["role"], "parts": [m["content"]]} for m in messages]
    response = client.generate_content(
        messages,
        generation_config=genai.types.GenerationConfig(
            candidate_count=n,
            max_output_tokens=max_new_tokens,
            temperature=temperature,
        ),
        safety_settings=[
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        ],
    )

    return response.text


def make_auto_request(*args, **kwargs) -> genai.types.GenerateContentResponse:
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        except ResourceExhausted as e:
            print("Rate limit exceeded. Waiting...", e.message)
            time.sleep(10)
        except GoogleAPICallError as e:
            print(e.message)
            time.sleep(1)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret

