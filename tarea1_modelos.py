"""
TAREA 1 - Clasificación con todas las variables
Modelos: Gaussian Naive Bayes  |  Decision Tree Classifier
"""

import os
import pickle
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data_processed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# =============================================================================
# 1. CARGA DE DATOS PROCESADOS
# =============================================================================
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv")).values
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv")).values
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

print("=" * 60)
print("TAREA 1 — Clasificación con todas las variables")
print("=" * 60)
print(f"Tamaño entrenamiento : {X_train.shape[0]} muestras")
print(f"Tamaño prueba        : {X_test.shape[0]} muestras")
print(f"Número de variables  : {X_train.shape[1]}")

# =============================================================================
# 2. FUNCIÓN DE EVALUACIÓN
# =============================================================================
CLASS_NAMES = ["Bajo (0)", "Medio (1)", "Alto (2)"]

def evaluar_modelo(nombre, modelo, y_real, y_pred):
    """Calcula métricas, las imprime en consola y devuelve un dict."""
    acc  = accuracy_score(y_real, y_pred)
    prec = precision_score(y_real, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_real, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_real, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_real, y_pred)

    print(f"\n{'=' * 60}")
    print(f"  MODELO: {nombre}")
    print(f"{'=' * 60}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f} %)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Matriz de Confusión:")
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.index.name   = "Real \\ Predicho"
    print(cm_df.to_string())
    print(f"\n  Reporte por clase:\n")
    print(classification_report(y_real, y_pred,
                                 target_names=CLASS_NAMES, zero_division=0))

    return {
        "modelo": nombre,
        "accuracy": round(acc, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
        "f1_weighted": round(f1, 4),
        "confusion_matrix": cm,
    }

# =============================================================================
# 3. MODELO 1 — GAUSSIAN NAIVE BAYES
# =============================================================================
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

resultados_gnb = evaluar_modelo("Gaussian Naive Bayes", gnb, y_test, y_pred_gnb)

# Guardar modelo entrenado
with open(os.path.join(MODELS_DIR, "naive_bayes.pkl"), "wb") as f:
    pickle.dump(gnb, f)

# =============================================================================
# 4. MODELO 2 — DECISION TREE CLASSIFIER
# =============================================================================
# max_depth=6 limita la profundidad para evitar sobreajuste (overfitting)
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

resultados_dt = evaluar_modelo("Decision Tree (max_depth=6)", dt, y_test, y_pred_dt)

# Guardar modelo entrenado
with open(os.path.join(MODELS_DIR, "decision_tree.pkl"), "wb") as f:
    pickle.dump(dt, f)

# =============================================================================
# 5. EXPORTAR RESULTADOS A /results
# =============================================================================
# --- 5a. CSV con métricas resumidas ---
resumen = pd.DataFrame([
    {
        "Modelo"             : resultados_gnb["modelo"],
        "Accuracy"           : resultados_gnb["accuracy"],
        "Precision (weighted)": resultados_gnb["precision_weighted"],
        "Recall (weighted)"  : resultados_gnb["recall_weighted"],
        "F1-Score (weighted)": resultados_gnb["f1_weighted"],
    },
    {
        "Modelo"             : resultados_dt["modelo"],
        "Accuracy"           : resultados_dt["accuracy"],
        "Precision (weighted)": resultados_dt["precision_weighted"],
        "Recall (weighted)"  : resultados_dt["recall_weighted"],
        "F1-Score (weighted)": resultados_dt["f1_weighted"],
    },
])
resumen.to_csv(os.path.join(RESULTS_DIR, "metricas_tarea1.csv"), index=False)

# --- 5b. TXT con reporte detallado (listo para presentación) ---
txt_path = os.path.join(RESULTS_DIR, "reporte_tarea1.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("TAREA 1 — RESULTADOS DE CLASIFICACIÓN\n")
    f.write("Dataset: StressLevelDataset.csv\n")
    f.write("Split: 70% entrenamiento / 30% prueba  |  random_state=42\n")
    f.write("=" * 60 + "\n\n")

    for res in [resultados_gnb, resultados_dt]:
        f.write(f"MODELO: {res['modelo']}\n")
        f.write(f"  Accuracy            : {res['accuracy']:.4f}  "
                f"({res['accuracy']*100:.2f} %)\n")
        f.write(f"  Precision (weighted): {res['precision_weighted']:.4f}\n")
        f.write(f"  Recall    (weighted): {res['recall_weighted']:.4f}\n")
        f.write(f"  F1-Score  (weighted): {res['f1_weighted']:.4f}\n")
        f.write("\n  Matriz de Confusión (filas=Real, columnas=Predicho):\n")
        cm_df = pd.DataFrame(
            res["confusion_matrix"],
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        )
        f.write(cm_df.to_string() + "\n")
        f.write("\n" + "-" * 60 + "\n\n")

print("\n" + "=" * 60)
print("RESULTADOS EXPORTADOS")
print(f"  {os.path.join(RESULTS_DIR, 'metricas_tarea1.csv')}")
print(f"  {os.path.join(RESULTS_DIR, 'reporte_tarea1.txt')}")
print("MODELOS GUARDADOS")
print(f"  {os.path.join(MODELS_DIR, 'naive_bayes.pkl')}")
print(f"  {os.path.join(MODELS_DIR, 'decision_tree.pkl')}")
print("=" * 60)
