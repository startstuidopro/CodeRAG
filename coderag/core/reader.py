from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import List, Dict

class ZephyrReader:
    def __init__(self, model_name="HuggingFaceH4/zephyr-7b-beta"):
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_prompt(self, context: str, question: str) -> str:
        return f"""<|system|>
Using the information contained in the context, give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.</s>
<|user|>
Context:
{context}
---
Now here is the question you need to answer.

Question: {question}</s>
<|assistant|>"""

    def generate_answer(self, context: str, question: str) -> str:
        prompt = self.format_prompt(context, question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=500,
            temperature=0.2,
            repetition_penalty=1.1,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def validate_response(self, response: str) -> bool:
        """Check for valid response structure and source citations"""
        return any([
            "cannot be deduced" in response,
            "source document" in response,
            len(response.strip()) > 10
        ])
