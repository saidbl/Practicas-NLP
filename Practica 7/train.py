!pip uninstall -y torchao
!pip install -q -U transformers datasets accelerate peft bitsandbytes sentencepiece
from google.colab import drive
drive.mount('/content/drive')

import os
import json
import torch
import shutil

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
)

dataset_path = "/content/drive/MyDrive/datasets/it_helpdesk_dataset_finetuning.json"

base_model_id = "unsloth/Llama-3.2-1B-Instruct"

lora_save_path = "/content/drive/MyDrive/it_helpdesk_llama_lora_adapter"

merged_save_path = "/content/drive/MyDrive/it_helpdesk_llama_lora_merged_model"

training_output_dir = "/content/it_helpdesk_llama_lora_training"

folders_to_delete = [
    lora_save_path,
    merged_save_path,
    training_output_dir
]

for folder in folders_to_delete:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print("Carpeta eliminada:", folder)

print("Limpieza completada.")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(
        "No se encontró el dataset en:\n"
        f"{dataset_path}"
    )

with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Dataset encontrado:", dataset_path)
print("Total de ejemplos:", len(data))
print("\nPrimer ejemplo:")
print(json.dumps(data[0], ensure_ascii=False, indent=2))

raw_dataset = load_dataset(
    "json",
    data_files={"train": dataset_path},
    download_mode="force_redownload"
)

train_dataset = raw_dataset["train"]

print("\nEjemplos usados para entrenamiento:", len(train_dataset))

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

print("Tokenizer cargado correctamente.")

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

base_model.config.use_cache = False

print("Modelo base cargado sin bitsandbytes:", base_model_id)

print("Modelo base cargado:", base_model_id)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)

print("\nParámetros entrenables:")
model.print_trainable_parameters()

system_message = (
    "You are an IT Helpdesk support chatbot. "
    "Always answer in English. "
    "Your answer must include a diagnosis and a clear solution. "
    "Use this exact format: Diagnosis: ... Solution: ..."
)

def build_chat_text(example):
    user_message = example["input"]
    assistant_message = example["output"]

    messages_full = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_message
        },
        {
            "role": "assistant",
            "content": assistant_message
        }
    ]

    messages_prompt = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    full_text = tokenizer.apply_chat_template(
        messages_full,
        tokenize=False,
        add_generation_prompt=False
    )

    prompt_text = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=False,
        add_generation_prompt=True
    )

    return {
        "full_text": full_text,
        "prompt_text": prompt_text
    }

formatted_dataset = train_dataset.map(build_chat_text)

print("\nEjemplo de entrenamiento:")
print(formatted_dataset[0]["full_text"])

max_length = 512

def tokenize_and_mask(example):
    full = tokenizer(
        example["full_text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

    prompt = tokenizer(
        example["prompt_text"],
        truncation=True,
        max_length=max_length,
        padding=False
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    labels = input_ids.copy()

    prompt_len = len(prompt["input_ids"])

    labels[:prompt_len] = [-100] * prompt_len

    labels = [
        token if mask == 1 else -100
        for token, mask in zip(labels, attention_mask)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_dataset = formatted_dataset.map(
    tokenize_and_mask,
    remove_columns=formatted_dataset.column_names
)

print("\nDataset tokenizado correctamente.")

training_args = TrainingArguments(
    output_dir=training_output_dir,

    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,

    num_train_epochs=6,
    learning_rate=2e-4,

    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,

    report_to="none",

    fp16=True,
    optim="adamw_torch",

    warmup_ratio=0.05,
    lr_scheduler_type="cosine",

    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

print("\nIniciando entrenamiento LoRA con Llama...")

trainer.train()

print("\nEntrenamiento finalizado.")

model.save_pretrained(lora_save_path)
tokenizer.save_pretrained(lora_save_path)

print("\nAdaptador LoRA guardado en:")
print(lora_save_path)

print("\nFusionando LoRA con modelo base...")

del model
del base_model
torch.cuda.empty_cache()

base_model_for_merge = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

merged_model = PeftModel.from_pretrained(
    base_model_for_merge,
    lora_save_path
)

merged_model = merged_model.merge_and_unload()

merged_model.save_pretrained(
    merged_save_path,
    safe_serialization=True
)

tokenizer.save_pretrained(merged_save_path)

print("\nModelo fusionado guardado en:")
print(merged_save_path)

print("\nProceso completo.")