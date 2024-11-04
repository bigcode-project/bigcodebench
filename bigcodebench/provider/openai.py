import os
from typing import List

import openai

from bigcodebench.gen.util.openai_request import make_auto_request
from bigcodebench.provider.utility import make_raw_chat_prompt
from bigcodebench.provider.base import DecoderBase
from bigcodebench.provider.utility import concurrent_call

class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, base_url=None, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "none"), base_url=base_url
        )

    # def codegen(
    #     self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    # ) -> List[str]:
    #     if do_sample:
    #         assert self.temperature > 0, "Temperature must be positive for sampling"
    #     all_outputs = []
    #     for prompt in tqdm(prompts):
    #         outputs = []
    #         message = make_raw_chat_prompt(
    #             task_prompt=prompt,
    #             subset=self.subset,
    #             split=self.split,
    #             instruction_prefix=self.instruction_prefix,
    #             response_prefix=self.response_prefix,
    #             tokenizer=None,
    #         )
    #         ret = make_auto_request(
    #             self.client,
    #             message=message,
    #             model=self.name,
    #             max_tokens=self.max_new_tokens,
    #             temperature=self.temperature,
    #             n=num_samples,
    #         )
    #         for item in ret.choices:
    #             outputs.append(item.message.content)
    #         all_outputs.append(outputs)
    #     return all_outputs

    # def is_direct_completion(self) -> bool:
    #     return False
    
    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        messages = [make_raw_chat_prompt(
            task_prompt=prompt,
            subset=self.subset,
            split=self.split,
            instruction_prefix=self.instruction_prefix,
            response_prefix=self.response_prefix,
            tokenizer=None,
        ) for prompt in prompts]
        # use concurrency based batching for o1 and deepseek models
        if self.name.startswith("o1-") or self.name == "deepseek-chat":
            return self._codegen_batch_via_concurrency(messages, num_samples)

        return self._codegen_api_batch(messages, num_samples)

    def _codegen_api_batch(self, messages: List[str], num_samples: int) -> List[str]:
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "none"), base_url=self.base_url
        )
        
        all_outputs = []
        for message in messages:
            ret = make_auto_request(
                client,
                message=message,
                model=self.name,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                n=num_samples,
            )
            outputs = []
            for item in ret.choices:
                outputs.append(item.message.content)
            all_outputs.append(outputs)
        return all_outputs

    def _codegen_batch_via_concurrency(self, messages: List[str], num_samples: int) -> List[str]:
        batches = concurrent_call(
            num_samples, self._codegen_api_batch, messages, num_samples=1
        )
        return [b[0] for b in batches]

    def is_direct_completion(self) -> bool:
        return False