from abc import ABC, abstractmethod

import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig

from openai import OpenAI
import os
import tiktoken


class LlmManager(ABC):
    """
    An "interface" for various LLM manager objects.
    """

    @abstractmethod
    def chat_completion(
        self,
        prompt,
        print_result=False,
        seed=42,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.0,
    ):
        pass


class HuggingFaceLlmManager(LlmManager):
    def __init__(
        self,
        model_name,
        cache_dir="/vol/bitbucket/clarg/argumentative-llm/cache",
        model_args=None,
        input_device="cuda:0",
        quantization="4bit",
    ):
        super().__init__()
        
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif quantization == "none":
            quantization_config = None
        else:
            raise ValueError(f"Invalid quantization value {quantization}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": quantization_config,
                "cache_dir": cache_dir,
            },
        )
        self.input_device = input_device

    def chat_completion(
        self,
        message,
        print_result=False,
        seed=42,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.0,
        constraint_prefix=None,
        constraint_options=None,
        constraint_end_after_options=False,
        trim_response=True,
        apply_template=True,
    ):
        transformers.set_seed(seed)
        messages = [{"role": "user", "content": message}]
        
        if apply_template:
            prompt = self.pipeline.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = message
        
        if constraint_prefix is not None or constraint_options is not None:
            prefix_allowed_tokens_fn = self.construct_constraint_fun(
                prompt,
                force_prefix=constraint_prefix,
                force_options=constraint_options,
                end_after_options=constraint_end_after_options,
            )
        else:
            prefix_allowed_tokens_fn = None
        
        response = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            pad_token_id=self.tokenizer.eos_token_id
        )[0]["generated_text"]

        if print_result:
            print(response, flush=True)

        if trim_response:
            response = response.replace(prompt, "").strip()

        return response

    def construct_constraint_fun(self, prompt, force_prefix=None, force_options=None, end_after_options=False):
        if force_prefix is not None:
            force_prefix = self.tokenizer(force_prefix).input_ids[1:]
        if force_options is not None:
            force_options = [self.tokenizer(op).input_ids[1:] for op in force_options]

        def constraint_fun(batch_id, input_ids):
            prompt_len = len(self.tokenizer(prompt).input_ids[1:])
            generated_tokens = input_ids[prompt_len:].tolist()
            num_generated = len(generated_tokens)
            prefix_len = 0 if force_prefix is None else len(force_prefix)

            if force_prefix is not None and num_generated < prefix_len:
                # Force prefix to be generated first if provided
                return [force_prefix[num_generated]]
            elif num_generated >= prefix_len and force_options is not None:
                # Determine what option tokens have been generated
                op_tokens = generated_tokens[prefix_len:]
                num_op = len(op_tokens)

                # Calculate valid option continuations
                possible_continuations = [
                    c[num_op] for c in force_options if num_op < len(c) and c[:num_op] == op_tokens
                ]

                if not possible_continuations and end_after_options:
                    # No further continuations — terminate generation as requested
                    return [self.tokenizer.eos_token_id]
                elif not possible_continuations:
                    # No further continuations, but can continue free generation
                    return list(range(self.tokenizer.vocab_size))
                else:
                    # Allow generation to terminate if desirable
                    if op_tokens in force_options:
                        possible_continuations.append(self.tokenizer.eos_token_id)
                    # Force generation according to options
                    return possible_continuations
            else:
                return list(range(self.tokenizer.vocab_size))

        return constraint_fun


class OpenAiLlmManager(LlmManager):
    def __init__(
        self,
        model_name,
    ):
        self.model_name = model_name.split("openai/")[1]
        self.enc = tiktoken.encoding_for_model(self.model_name)
        self.client = OpenAI(api_key=os.environ["OPENAI_KEY"])

    def chat_completion(
        self,
        message,
        print_result=False,
        seed=42,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.0,
        constraint_prefix=None,
        constraint_options=None,
        constraint_end_after_options=False,
        trim_response=True,
        apply_template=True
    ):
        prompt = message + (("\n\n" + constraint_prefix) if constraint_prefix else "")

        logit_bias_map = {}
        for x in constraint_options:
            logit_bias_map[str(self.enc.encode(x)[0])] = 100

        tools = [{
            "type": "function",
            "function": {
                "name": "get_relation",
                "description": "Get relation between the two arguments.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relation": {
                            "type": "string",
                            "enum": ["Support", "Attack", "No"],
                            "description": "The relation between the two arguments.",
                        },
                    },
                    "required": ["relation"],
                },
            }
        }]

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": prompt
                }],
            temperature=0, #temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
            seed=seed,
            presence_penalty=repetition_penalty,
            # stop=['#'],  # stops after generating a new line, however does not return the stop sequence
            logit_bias=logit_bias_map,  # gives a better chance for these tokens to appear in the output
            logprobs=True,
            top_logprobs=20,
            #tools=tools,
            #tool_choice={"type": "function", "function": {"name": "get_relation"}}
        )

        response = completion.choices[0].message.content
        logits = completion.choices[0].logprobs.content[0].top_logprobs

        if print_result:
            print(prompt, response, flush=True)

        if trim_response:
            response = response.replace(prompt, "").strip()

        return response, logits
