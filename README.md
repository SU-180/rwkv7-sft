# RWKV7-G1j-13.3B QLoRA SFT Fine-tuning

![RWKV](https://img.shields.io/badge/RWKV7-G1j-blue)
![QLoRA](https://img.shields.io/badge/Training-QLoRA-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red)
![CUDA](https://img.shields.io/badge/CUDA-11.8-orange)


This project implements **QLoRA supervised fine-tuning (SFT)** on the
**RWKV7-G1j-13.3B-BF16** base model.

The goal is to transform the RWKV7 base model into an
instruction-following conversational model and provide a foundation
for further **Agent / Tool Calling / DSH integration**.


---

# 1. Project Overview


Large Language Models usually require post-training to improve:

- Instruction following
- Conversation ability
- Task understanding
- Agent capability


This project uses:

```
RWKV7-G1j-13.3B-BF16
                |
                |
                ↓

          QLoRA SFT

                |
                |
                ↓

       RWKV7 Instruction Model

                |
                |
                ↓

      Agent / Tool Calling SFT

                |
                |
                ↓

              DSH
```


---

# 2. Features


## Supported

✅ RWKV7 HuggingFace model loading

✅ 4bit NF4 QLoRA training

✅ LoRA parameter-efficient fine-tuning

✅ Multi-GPU training with Accelerate + DDP

✅ BF16 mixed precision training

✅ Adapter-based inference


## Future Extension

- Tool Calling SFT
- Function Calling
- Agent planning
- GGUF conversion
- Ollama deployment
- DSH integration


---

# 3. Model Information


## Base Model

```
RWKV7-G1j-13.3B-BF16
```


Architecture:

```
RWKV7
```


Parameters:

```
13.3B
```


Precision:

```
BF16
```


Model format:

```
HuggingFace Transformers
```


---

# 4. QLoRA Training Principle


QLoRA combines:

- 4bit quantization
- LoRA adapter training


Instead of updating all model parameters:


```
Original:

RWKV7 13.3B
     |
     ↓
Update all parameters


```


QLoRA:


```
RWKV7 Base

(4bit NF4, frozen)

        +

LoRA Adapter

(trainable)

        ↓

Updated Model
```


Advantages:

- Lower GPU memory usage
- Faster training
- Only a small number of parameters are updated


---

# 5. Training Environment


## Hardware


Tested on:


```
GPU:

NVIDIA RTX3090 × 2


VRAM:

24GB × 2
```



---

## Software


Operating System:

```
Ubuntu Linux
```


Python:

```
Python 3.11
```


CUDA:

```
CUDA 11.8
```


PyTorch:

```
torch 2.7.1+cu118
```


Main dependencies:


```
transformers==5.17.0

trl==1.13.0

peft==0.20.0

accelerate

bitsandbytes

datasets
```


---

# 6. Project Structure


```
rwkv7_qlora_sft/

├── configs/
│
│   └── training configuration
│
├── data/
│
│   └── SFT training dataset
│
├── scripts/
│
│   ├── train_rwkv7_qlora.py
│   │
│   └── test_adapter.py
│
├── output/
│
│   └── adapter/
│
│       ├── adapter_model.safetensors
│       └── adapter_config.json
│
├── run_train.sh
│
└── README.md

```


---

# 7. Dataset Format


Current SFT format:


```json
{
    "instruction":
    "介绍一下RWKV模型",

    "output":
    "RWKV是一种结合RNN和Transformer优点的语言模型..."
}
```


Training objective:


```
Instruction

        ↓

Model Understanding

        ↓

Response Generation
```


---

# 8. Training


## Single GPU


```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train_rwkv7_qlora.py
```


---

## Multi GPU


RTX3090 ×2:


```bash
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch \
--num_processes=2 \
--mixed_precision=bf16 \
scripts/train_rwkv7_qlora.py
```


Training uses:


```
Accelerate

+

Distributed Data Parallel (DDP)

```


---

# 9. Training Result


Training successfully completed.


Trainable parameters:


```
71,958,528
```


Total parameters:


```
13,341,204,480
```


Trainable ratio:


```
0.5394%
```


Loss:


```
Epoch 1:
1.443


Epoch 2:
0.8186


Epoch 3:
0.4759
```


---

# 10. Output Adapter


After training:


```
output/

└── adapter/

    ├── adapter_model.safetensors

    ├── adapter_config.json

    └── tokenizer files

```


The adapter contains:

```
LoRA incremental weights
```


The original RWKV7 model is not modified.


---

# 11. Adapter Testing


Run:


```bash
PYTHONNOUSERSITE=1 \
python scripts/test_adapter.py
```


Example:


Input:


```
介绍一下RWKV模型
```


Output:


```
RWKV是一种结合RNN和Transformer优点的语言模型...
```


The test confirms:

- LoRA adapter loading
- Instruction format learning
- SFT capability


---

# 12. Roadmap


## Phase 1 ✅

RWKV7 Base

↓

QLoRA Instruction SFT

↓

RWKV7 Chat Model



## Phase 2 🚧


Tool Calling SFT


```
User Request

      ↓

Intent Understanding

      ↓

Function Selection

      ↓

JSON Tool Call

```



## Phase 3


Deployment:


```
RWKV7-Agent

        ↓

GGUF

        ↓

Ollama

        ↓

DSH
```


---

# 13. Author


GitHub:

```
SU-180
```


---

# 14. License


This project follows the license of the original RWKV model.

```
Apache License 2.0
```

