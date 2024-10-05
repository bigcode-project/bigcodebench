import os
from typing import List
from tqdm import tqdm

import openai

from bigcodebench.provider.base import DecoderBase
from bigcodebench.gen.util.openai_request import make_auto_request
from bigcodebench.provider.utility import make_raw_chat_prompt

class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, base_url=None, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "none"), base_url=base_url
        )

    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        all_outputs = []
        for prompt in tqdm(prompts):
            outputs = []
            message = make_raw_chat_prompt(
                task_prompt=prompt,
                subset=self.subset,
                split=self.split,
                instruction_prefix=self.instruction_prefix,
                response_prefix=self.response_prefix,
                tokenizer=None,
            )
            ret = make_auto_request(
                self.client,
                message=message,
                model=self.name,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                n=num_samples,
            )
            for item in ret.choices:
                outputs.append(item.message.content)
            all_outputs.append(outputs)
        return all_outputs

    def is_direct_completion(self) -> bool:
        return False