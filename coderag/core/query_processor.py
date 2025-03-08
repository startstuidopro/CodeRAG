from transformers import pipeline
from langchain.schema import Document
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class QueryProcessor:
    def __init__(self, model_name="HuggingFaceH4/zephyr-7b-beta"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config
        )
        
        self.llm = pipeline(
            model=model,
            tokenizer=self.tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        )

    @track_performance
    def build_prompt(self, question: str, context: List[Document]) -> str:
        context_str = "\nExtracted documents:\n" + "\n".join(
            [f"Document {i}:::\n{doc.page_content}" for i, doc in enumerate(context)]
        )
        
        return self.tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "Using the information in the context, answer the question concisely. Cite document numbers when relevant.",
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context_str}\n---\nQuestion: {question}"
                }
            ], 
            tokenize=False, 
            add_generation_prompt=True
        )
