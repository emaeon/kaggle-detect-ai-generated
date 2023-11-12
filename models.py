from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
import torch

#TO-DO: model compression like LoRA, Quantization

def load_model_and_tokenizer(model_name):
        
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map='auto',
                                                torch_dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

