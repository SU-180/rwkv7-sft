#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test RWKV7 QLoRA Adapter

Base:
RWKV7-G1j-13.3B-BF16

Adapter:
rwkv7_qlora_sft/output/adapter

Purpose:
Verify whether SFT capability is learned.
"""


import os

# ================================
# Environment
# ================================

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["PYTHONNOUSERSITE"] = "1"


import torch


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


from peft import (
    PeftModel
)



# ================================
# Path
# ================================


BASE_MODEL = (
    "/home/dxl/RWKV/models/"
    "RWKV7-G1j-13.3B-BF16"
)


ADAPTER_PATH = (
    "/home/dxl/RWKV/"
    "rwkv7_qlora_sft/output/adapter"
)



# ================================
# Quantization
# ================================


bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True

)



# ================================
# Load tokenizer
# ================================


print("Loading tokenizer...")


tokenizer = AutoTokenizer.from_pretrained(

    BASE_MODEL,

    trust_remote_code=True

)


tokenizer.padding_side="right"



# ================================
# Load Base Model
# ================================


print("Loading base RWKV7...")


model = AutoModelForCausalLM.from_pretrained(

    BASE_MODEL,

    trust_remote_code=True,

    quantization_config=bnb_config,

    dtype=torch.bfloat16,

    device_map="auto"

)



print("Loading LoRA adapter...")


model = PeftModel.from_pretrained(

    model,

    ADAPTER_PATH

)



model.eval()


print("RWKV7-Agent loaded!")



# ================================
# Test Function
# ================================


def chat(prompt):


    text = f"""
### Instruction:
{prompt}

### Response:
"""


    inputs = tokenizer(

        text,

        return_tensors="pt"

    ).to(model.device)



    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=256,

            temperature=0.7,

            top_p=0.9,

            do_sample=True,

            repetition_penalty=1.1

        )


    result = tokenizer.decode(

        outputs[0],

        skip_special_tokens=True

    )


    return result



# ================================
# Test Cases
# ================================


tests=[

    "你是谁？",

    "介绍一下RWKV模型。",

    "什么是LoRA微调？",

    "如果用户要求查询天气，你应该如何处理？"

]



print("\n==============================")
print("RWKV7-Agent SFT Test")
print("==============================\n")



for q in tests:


    print("--------------------------------")
    print("User:")
    print(q)


    answer = chat(q)


    print("\nModel:")
    print(answer)


print("\nDONE")

