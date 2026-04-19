import os
import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
USE_TEXT_CLEANING = True
OUTPUT_DIR = "resultados_amazon_polarity_full_mejorado"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TFIDF_CONFIG = {
    "max_features": 20000,
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 0.90,
    "sublinear_tf": True,
    "strip_accents": "unicode",
    "lowercase": True,
    "stop_words": "english"
}

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9.,!?;:'\"()\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_texts(texts, use_cleaning=True):
    if use_cleaning:
        return [clean_text(t) for t in texts]
    return list(texts)


def save_dataframe(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def print_section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def compute_metrics(y_true, y_pred, y_score=None, y_proba=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    if y_score is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_score)
        metrics["PR-AUC"] = average_precision_score(y_true, y_score)
    else:
        metrics["ROC-AUC"] = np.nan
        metrics["PR-AUC"] = np.nan

    if y_proba is not None:
        metrics["Log Loss"] = log_loss(y_true, y_proba)
    else:
        metrics["Log Loss"] = np.nan

    return metrics


def get_prediction_outputs(model, X_test):
    y_pred = model.predict(X_test)

    y_score = None
    y_proba = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        y_proba = proba[:, 1]
        y_score = y_proba
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)

    return y_pred, y_score, y_proba


def save_confusion_matrix(cm, model_name):
    cm_df = pd.DataFrame(
        cm,
        index=["Real Negativa (0)", "Real Positiva (1)"],
        columns=["Pred Negativa (0)", "Pred Positiva (1)"]
    )
    filename = f"confusion_matrix_{model_name}.csv"
    save_dataframe(cm_df.reset_index().rename(columns={"index": "Clase Real"}), filename)
    return cm_df


def plot_roc_curve(y_true, y_score, model_name):
    if y_score is None:
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Curva ROC - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"roc_curve_{model_name}.png"), dpi=200)
    plt.close()


def plot_pr_curve(y_true, y_score, model_name):
    if y_score is None:
        return

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Curva Precision-Recall - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"pr_curve_{model_name}.png"), dpi=200)
    plt.close()

def classification_report_to_df(y_true, y_pred):
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    df = pd.DataFrame(report).transpose().reset_index()
    df = df.rename(columns={"index": "Clase/Métrica"})
    return df

def show_dataset_distribution(y, split_name):
    counts = pd.Series(y).value_counts().sort_index()
    df = pd.DataFrame({
        "Split": split_name,
        "Clase": counts.index,
        "Cantidad": counts.values,
        "Porcentaje": (counts.values / len(y) * 100).round(2)
    })
    return df

print_section("1. CARGA DEL DATASET COMPLETO")

start_total = time.time()

print("Cargando amazon_polarity completo...")
dataset = load_dataset("amazon_polarity")

X_train_raw = dataset["train"]["content"]
y_train = np.array(dataset["train"]["label"], dtype=int)

X_test_raw = dataset["test"]["content"]
y_test = np.array(dataset["test"]["label"], dtype=int)

print(f"Tamaño train: {len(X_train_raw):,}")
print(f"Tamaño test : {len(X_test_raw):,}")

dist_train = show_dataset_distribution(y_train, "train")
dist_test = show_dataset_distribution(y_test, "test")
dist_df = pd.concat([dist_train, dist_test], ignore_index=True)

print("\nDistribución de clases:")
print(dist_df.to_string(index=False))

save_dataframe(dist_df, "distribucion_clases.csv")

print_section("2. PREPROCESAMIENTO")

print("Aplicando limpieza de texto...")
t0 = time.time()

X_train = preprocess_texts(X_train_raw, USE_TEXT_CLEANING)
X_test = preprocess_texts(X_test_raw, USE_TEXT_CLEANING)

t1 = time.time()
print(f"Preprocesamiento completado en {(t1 - t0)/60:.2f} minutos")

print_section("3. DISEÑO EXPERIMENTAL")

print("Problema: clasificación binaria de reseñas de Amazon")
print("Clase 0 = negativa")
print("Clase 1 = positiva")
print("Se usa TF-IDF mejorado para dataset completo:")
for k, v in TFIDF_CONFIG.items():
    print(f" - {k}: {v}")

print("\nModelos comparados:")
print("1) SGDClassifier con pérdida logística")
print("2) Complement Naive Bayes")

print_section("4. DEFINICIÓN DE MODELOS")

models = {
    "SGDClassifier_LogLoss_Mejorado": Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-6,
            max_iter=50,
            tol=1e-3,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=3,
            random_state=RANDOM_STATE
        ))
    ]),

    "ComplementNB_Mejorado": Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
        ("clf", ComplementNB(alpha=0.3))
    ])
}

for model_name in models.keys():
    print(f"- {model_name}")

print_section("5. ENTRENAMIENTO Y EVALUACIÓN")

all_results = []
all_predictions = {}

for model_name, model in models.items():
    print(f"\nEntrenando: {model_name}")
    train_start = time.time()

    model.fit(X_train, y_train)

    train_end = time.time()
    print(f"Tiempo de entrenamiento: {(train_end - train_start)/60:.2f} minutos")

    print(f"Evaluando: {model_name}")
    eval_start = time.time()

    y_pred, y_score, y_proba = get_prediction_outputs(model, X_test)

    eval_end = time.time()
    print(f"Tiempo de evaluación: {(eval_end - eval_start)/60:.2f} minutos")

    metrics = compute_metrics(y_test, y_pred, y_score, y_proba)
    metrics["Modelo"] = model_name
    metrics["Train Time (min)"] = round((train_end - train_start) / 60, 2)
    metrics["Eval Time (min)"] = round((eval_end - eval_start) / 60, 2)

    all_results.append(metrics)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = save_confusion_matrix(cm, model_name)

    report_df = classification_report_to_df(y_test, y_pred)
    save_dataframe(report_df, f"classification_report_{model_name}.csv")

    plot_roc_curve(y_test, y_score, model_name)
    plot_pr_curve(y_test, y_score, model_name)

    all_predictions[model_name] = {
        "model": model,
        "y_pred": y_pred,
        "y_score": y_score,
        "y_proba": y_proba,
        "confusion_matrix": cm,
        "confusion_matrix_df": cm_df,
        "classification_report_df": report_df
    }

    print("\nMétricas:")
    for key, value in metrics.items():
        if key != "Modelo":
            if isinstance(value, float) and not np.isnan(value):
                print(f"{key:20}: {value:.4f}")
            else:
                print(f"{key:20}: {value}")

    print("\nMatriz de confusión:")
    print(cm_df.to_string())

    print("\nClassification Report:")
    print(report_df.round(4).to_string(index=False))

print_section("6. TABLA COMPARATIVA FINAL")

results_df = pd.DataFrame(all_results)

column_order = [
    "Modelo",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "Balanced Accuracy",
    "MCC",
    "ROC-AUC",
    "PR-AUC",
    "Log Loss",
    "Train Time (min)",
    "Eval Time (min)"
]
results_df = results_df[column_order]
results_df = results_df.sort_values(by=["F1-score", "ROC-AUC", "Accuracy"], ascending=False)

results_df_rounded = results_df.copy()
for col in results_df_rounded.columns:
    if col != "Modelo":
        results_df_rounded[col] = results_df_rounded[col].round(4)

print(results_df_rounded.to_string(index=False))
save_dataframe(results_df_rounded, "comparacion_modelos.csv")

print("\nTabla lista para copiar al reporte (formato texto):")
print(results_df_rounded.to_string(index=False))

print_section("7. SELECCIÓN DEL MEJOR MODELO")

best_model_name = results_df.iloc[0]["Modelo"]
best_info = all_predictions[best_model_name]

print(f"Mejor modelo según F1-score: {best_model_name}")
print("\nMatriz de confusión del mejor modelo:")
print(best_info["confusion_matrix_df"].to_string())

print("\nClassification report del mejor modelo:")
print(best_info["classification_report_df"].round(4).to_string(index=False))

print_section("8. ANÁLISIS DE ERRORES DEL MEJOR MODELO")

error_df = pd.DataFrame({
    "text": X_test,
    "y_true": y_test,
    "y_pred": best_info["y_pred"]
})

if best_info["y_score"] is not None:
    error_df["score"] = best_info["y_score"]

errors_only = error_df[error_df["y_true"] != error_df["y_pred"]].copy()

if "score" in errors_only.columns:
    errors_only["abs_score"] = errors_only["score"].abs()
    errors_only = errors_only.sort_values(by="abs_score", ascending=False)

errors_to_save = errors_only.head(100).copy()
save_dataframe(errors_to_save, "errores_mejor_modelo_top100.csv")

print(f"Cantidad de errores del mejor modelo: {len(errors_only):,}")
print("\nPrimeros 15 ejemplos mal clasificados:")

for i, row in errors_only.head(15).iterrows():
    text_short = row["text"][:300].replace("\n", " ")
    print("-" * 80)
    print(f"Índice: {i}")
    print(f"Real: {row['y_true']} | Predicho: {row['y_pred']}")
    if "score" in row:
        print(f"Score: {row['score']:.4f}")
    print(f"Texto: {text_short}...")

print_section("9. TABLA RESUMEN DEL EXPERIMENTO")

summary_df = pd.DataFrame({
    "Aspecto": [
        "Dataset",
        "Tipo de problema",
        "Clase negativa",
        "Clase positiva",
        "Tamaño train",
        "Tamaño test",
        "Representación",
        "Configuración TF-IDF",
        "Modelos comparados",
        "Uso de todo el dataset",
        "Limpieza de texto"
    ],
    "Valor": [
        "amazon_polarity",
        "Clasificación binaria",
        "0 = negativa",
        "1 = positiva",
        f"{len(X_train):,}",
        f"{len(X_test):,}",
        "TF-IDF",
        "max_features=20000, ngram_range=(1,2), stop_words='english'",
        "SGDClassifier_LogLoss_Mejorado y ComplementNB_Mejorado",
        "Sí",
        "Sí" if USE_TEXT_CLEANING else "No"
    ]
})

print(summary_df.to_string(index=False))
save_dataframe(summary_df, "resumen_experimento.csv")

print_section("10. CONCLUSIÓN AUTOMÁTICA")
print(f"El mejor modelo fue: {best_model_name}")

end_total = time.time()
print(f"\nTiempo total aproximado: {(end_total - start_total)/60:.2f} minutos")