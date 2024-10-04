import os
from typing import List

import google.generativeai as genai


from bigcodebench.provider.base import DecoderBase
from bigcodebench.gen.util.google_request import make_auto_request
from bigcodebench.provider.utility import make_raw_chat_prompt


class GoogleDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(name)

    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        
        all_outputs = []
        
        for prompt in tqdm(prompts):
            ret_texts = []
            message = make_raw_chat_prompt(
                task_prompt=prompt,
                subset=self.subset,
                split=self.split,
                instruction_prefix=self.instruction_prefix,
                response_prefix=self.response_prefix,
                tokenizer=None,
            )
            replies = make_auto_request(
                self.client,
                message,
                self.name,
                n=batch_size,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            for candidate in replies.candidates:
                parts = candidate.content.parts
                if parts:
                    ret_texts.append(parts[0].text)
                else:
                    print("Empty response!")
                    ret_texts.append("")
                    print(f"{candidate.safety_ratings = }")
            ret_texts.append("")
            all_outputs.append(ret_texts + [""] * (batch_size - len(ret_texts)))

        return all_outputs

    def is_direct_completion(self) -> bool:
        return False