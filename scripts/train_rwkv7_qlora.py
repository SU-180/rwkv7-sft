#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RWKV7-G1j-13.3B QLoRA SFT

Pipeline:

RWKV7 BF16 HF
        |
        ↓
4bit NF4
        |
        ↓
LoRA
        |
        ↓
SFT
        |
        ↓
Adapter

Multi GPU:
accelerate launch --num_processes=2
"""


import os


# ===============================
# Environment
# ===============================

# disable unused backend
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

# avoid ~/.local pollution
os.environ["PYTHONNOUSERSITE"] = "1"

# disable trl chunked CE patch
os.environ["TRL_DISABLE_CHUNKED_LOSS"] = "1"


import torch

from datasets import load_dataset


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)


from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)


from trl import SFTTrainer



# =====================================================
# Path
# =====================================================


MODEL_PATH = (
    "/home/dxl/RWKV/models/"
    "RWKV7-G1j-13.3B-BF16"
)


DATA_PATH = (
    "/home/dxl/RWKV/"
    "rwkv7_qlora_sft/data/train.jsonl"
)


OUTPUT_DIR = (
    "/home/dxl/RWKV/"
    "rwkv7_qlora_sft/output/adapter"
)



# =====================================================
# Quantization
# =====================================================


bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True

)



# =====================================================
# Tokenizer
# =====================================================


tokenizer = AutoTokenizer.from_pretrained(

    MODEL_PATH,

    trust_remote_code=True

)


# RWKV建议右padding
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token


# =====================================================
# Model
# =====================================================

print("Loading RWKV7 model...")


model = AutoModelForCausalLM.from_pretrained(

    MODEL_PATH,

    trust_remote_code=True,

    quantization_config=bnb_config,

    dtype=torch.bfloat16

)


print("Model loaded")



# =====================================================
# Prepare k-bit training
# =====================================================


model.gradient_checkpointing_enable()


model = prepare_model_for_kbit_training(
    model
)



# =====================================================
# LoRA
# =====================================================


lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    target_modules=[

        "receptance",
        "key",
        "value",
        "output"

    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"

)



model = get_peft_model(

    model,

    lora_config

)



model.print_trainable_parameters()



# =====================================================
# Dataset
# =====================================================


dataset = load_dataset(

    "json",

    data_files=DATA_PATH

)["train"]



def formatting_func(example):

    instruction = example.get(
        "instruction",
        ""
    )

    output = example.get(
        "output",
        ""
    )


    text = f"""### Instruction:
{instruction}

### Response:
{output}"""


    return [text]



# =====================================================
# Training
# =====================================================


training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,


    per_device_train_batch_size=1,


    gradient_accumulation_steps=8,


    learning_rate=2e-4,


    num_train_epochs=3,


    bf16=True,


    logging_steps=1,


    save_steps=50,


    save_total_limit=2,


    gradient_checkpointing=True,


    report_to="none"


)



trainer = SFTTrainer(

    model=model,
    

    args=training_args,

    train_dataset=dataset,

    processing_class=tokenizer,

    formatting_func=formatting_func,

    max_seq_length=2048
)



# =====================================================
# Train
# =====================================================


print("==============================")
print("Start Training")
print("==============================")


trainer.train()



# =====================================================
# Save LoRA
# =====================================================


print("Saving adapter...")


model.save_pretrained(
    OUTPUT_DIR
)


tokenizer.save_pretrained(
    OUTPUT_DIR
)


print("DONE")
