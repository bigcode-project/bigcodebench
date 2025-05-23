import signal
import time

import anthropic
from anthropic.types import Message


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def make_auto_request(client: anthropic.Client, *args, **kwargs) -> Message:
    ret = None
    while ret is None:
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(100)
            if "reasoning_budget" in kwargs and "reasoning_beta" in kwargs:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": kwargs["reasoning_budget"],
                }
                kwargs["betas"] = [kwargs["reasoning_beta"]]
                kwargs.pop("reasoning_budget")
                kwargs.pop("reasoning_beta")
                kwargs.pop("temperature")
            if "thinking" in kwargs:
                ret = client.beta.messages.create(*args, **kwargs, stream=True)
            else:
                ret = client.messages.create(*args, **kwargs)
            signal.alarm(0)
        except anthropic.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            signal.alarm(0)
            time.sleep(5)
        except anthropic.APIConnectionError:
            print("API connection error. Waiting...")
            signal.alarm(0)
            time.sleep(5)
        except anthropic.InternalServerError:
            print("Internal server error. Waiting...")
            signal.alarm(0)
            time.sleep(5)
        except anthropic.APIError as e:
            print("Unknown API error")
            print(e)
            if (
                e.body["error"]["message"]
                == "Output blocked by content filtering policy"
            ):
                raise Exception("Content filtering policy blocked output")
            signal.alarm(0)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            signal.alarm(0)
            time.sleep(1)
    return ret