!pip install -U transformers datasets peft trl accelerate bitsandbytes
import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# RUTAS
from google.colab import drive
drive.mount('/content/drive')
base_dir = "/content/drive/MyDrive/bash_command_helper"
dataset_dir = f"{base_dir}/dataset"
output_dir = f"{base_dir}/models/lora"

os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

dataset_path = f"{dataset_dir}/extended_bash_commands_dataset_ft.json"

# MODELO BASE

base_model_id = "unsloth/Llama-3.2-1B-Instruct"

# CARGAR DATASET

with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

def format_example(item):
    return {
        "text": (
            "You are a Linux command generator. "
            "Output only the command, no explanation.\n\n"
            f"User: {item['input']}\n"
            f"Assistant: {item['output']}"
        )
    }

dataset = Dataset.from_list([format_example(x) for x in data])

print(dataset[0])

# TOKENIZER Y MODELO

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    dtype=torch.float16,
    device_map="auto"
)

model.config.use_cache = False

# CONFIGURACIÓN LORA

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "v_proj",
    ]
)

# CONFIGURACIÓN ENTRENAMIENTO

sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    report_to="none",
    max_length=256,
    dataset_text_field="text"
)

# TRAINER

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config
)

# ENTRENAR

trainer.train()

# GUARDAR EN DRIVE

trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Modelo LoRA guardado en:", output_dir)