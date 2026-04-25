
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(4)  
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

SEED = 42
MODEL_NAME = "prajjwal1/bert-tiny"

np.random.seed(SEED)
torch.manual_seed(SEED)

dataset = load_dataset("amazon_polarity")

train_ds = dataset["train"].shuffle(seed=SEED).select(range(30000))
test_ds = dataset["test"].shuffle(seed=SEED).select(range(5000))

split = train_ds.train_test_split(test_size=0.2, seed=SEED)
train_ds = split["train"]
val_ds = split["test"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["content"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("label", "labels")
val_ds = val_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

training_args = TrainingArguments(
    output_dir="./amazon_bert_tiny_results",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    seed=SEED,
    report_to="none",
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    load_best_model_at_end=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

results = trainer.evaluate(test_ds)

print("\n=== RESULTADOS FINALES EN TEST ===")
for key, value in results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

predictions = trainer.predict(test_ds)

y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print("\n=== REPORTE DE CLASIFICACIÓN ===")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Negative", "Positive"]
))

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

disp.plot()
plt.title("Matriz de Confusión - BERT Tiny")
plt.show()

trainer.save_model("./modelo_amazon_bert_tiny")
tokenizer.save_pretrained("./modelo_amazon_bert_tiny")

texts = [
    "This product is fantastic and works perfectly.",
    "Terrible quality. It broke after one day.",
    "The item is acceptable, but not great."
]

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {key: value.to(device) for key, value in inputs.items()}

model.eval()

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    preds = torch.argmax(probs, dim=1)

print("\n=== PREDICCIONES DE EJEMPLO ===")
for text, pred, prob in zip(texts, preds, probs):
    label = "Positive" if pred.item() == 1 else "Negative"
    confidence = torch.max(prob).item()
    print(f"\nTexto: {text}")
    print(f"Predicción: {label}")
    print(f"Confianza: {confidence:.4f}")