"""
TAREA 2 - Clasificación con selección de características por Información Mutua (MMI)
Modelos: Gaussian Naive Bayes  |  Decision Tree Classifier
"""

import os
import pickle
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
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

# Resultados de Tarea 1 para la comparación final
T1_METRICS = {
    "Gaussian Naive Bayes":        {"accuracy": 0.8818, "f1": 0.8837},
    "Decision Tree (max_depth=6)": {"accuracy": 0.8848, "f1": 0.8849},
}

TOP_K = 10   # número de mejores características a seleccionar

# =============================================================================
# 1. CARGA DE DATOS PROCESADOS
# =============================================================================
X_train_df = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test_df  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train    = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test     = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

feature_names = X_train_df.columns.tolist()
X_train_full  = X_train_df.values
X_test_full   = X_test_df.values

print("=" * 60)
print("TAREA 2 — Selección por Información Mutua (MMI)")
print("=" * 60)
print(f"Características disponibles : {len(feature_names)}")
print(f"Top K seleccionado          : {TOP_K}")

# =============================================================================
# 2. CÁLCULO DE INFORMACIÓN MUTUA
# =============================================================================
mi_scores = mutual_info_classif(X_train_full, y_train, random_state=42)

ranking = pd.DataFrame({
    "caracteristica": feature_names,
    "mi_score":       mi_scores,
}).sort_values("mi_score", ascending=False).reset_index(drop=True)

ranking.index += 1   # ranking empieza en 1

print("\n--- RANKING COMPLETO POR INFORMACIÓN MUTUA ---")
print(ranking.to_string())

# Guardar ranking en CSV
ranking_path = os.path.join(RESULTS_DIR, "ranking_mmi.csv")
ranking.to_csv(ranking_path)
print(f"\nRanking guardado en: {ranking_path}")

# =============================================================================
# 3. SELECCIÓN DE LAS TOP K CARACTERÍSTICAS
# =============================================================================
top_features = ranking["caracteristica"].head(TOP_K).tolist()
top_idx      = [feature_names.index(f) for f in top_features]

X_train_sel = X_train_full[:, top_idx]
X_test_sel  = X_test_full[:, top_idx]

print(f"\n--- TOP {TOP_K} CARACTERÍSTICAS SELECCIONADAS ---")
for i, feat in enumerate(top_features, 1):
    score = ranking.loc[ranking["caracteristica"] == feat, "mi_score"].values[0]
    print(f"  {i:2d}. {feat:<35} score={score:.4f}")

# =============================================================================
# 4. FUNCIÓN DE EVALUACIÓN (reutilizable)
# =============================================================================
CLASS_NAMES = ["Bajo (0)", "Medio (1)", "Alto (2)"]

def evaluar_modelo(nombre, y_real, y_pred):
    """Calcula métricas, las imprime en consola y devuelve un dict."""
    acc  = accuracy_score(y_real, y_pred)
    prec = precision_score(y_real, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_real, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_real, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_real, y_pred)

    print(f"\n{'=' * 60}")
    print(f"  MODELO: {nombre}  [MMI Top-{TOP_K}]")
    print(f"{'=' * 60}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f} %)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Matriz de Confusión:")
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.index.name = "Real \\ Predicho"
    print(cm_df.to_string())
    print(f"\n  Reporte por clase:\n")
    print(classification_report(y_real, y_pred,
                                 target_names=CLASS_NAMES, zero_division=0))
    return {
        "modelo":    nombre,
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "cm":        cm,
    }

# =============================================================================
# 5. MODELO 1 — GAUSSIAN NAIVE BAYES  (MMI)
# =============================================================================
gnb_mmi = GaussianNB()
gnb_mmi.fit(X_train_sel, y_train)
y_pred_gnb = gnb_mmi.predict(X_test_sel)

res_gnb = evaluar_modelo("Gaussian Naive Bayes", y_test, y_pred_gnb)

with open(os.path.join(MODELS_DIR, "naive_bayes_mmi.pkl"), "wb") as f:
    pickle.dump(gnb_mmi, f)

# =============================================================================
# 6. MODELO 2 — DECISION TREE  (MMI)
# =============================================================================
dt_mmi = DecisionTreeClassifier(max_depth=6, random_state=42)
dt_mmi.fit(X_train_sel, y_train)
y_pred_dt = dt_mmi.predict(X_test_sel)

res_dt = evaluar_modelo("Decision Tree (max_depth=6)", y_test, y_pred_dt)

with open(os.path.join(MODELS_DIR, "decision_tree_mmi.pkl"), "wb") as f:
    pickle.dump(dt_mmi, f)

# =============================================================================
# 7. REPORTE COMPARATIVO TAREA 1 vs TAREA 2
# =============================================================================
txt_path = os.path.join(RESULTS_DIR, "reporte_tarea2.txt")

with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("TAREA 2 — SELECCIÓN POR INFORMACIÓN MUTUA (MMI)\n")
    f.write(f"Top-K seleccionado: {TOP_K} características\n")
    f.write("=" * 60 + "\n\n")

    # Ranking
    f.write("RANKING DE INFORMACIÓN MUTUA\n")
    f.write(ranking.to_string() + "\n\n")

    # Características seleccionadas
    f.write(f"TOP {TOP_K} CARACTERÍSTICAS USADAS\n")
    for i, feat in enumerate(top_features, 1):
        score = ranking.loc[ranking["caracteristica"] == feat, "mi_score"].values[0]
        f.write(f"  {i:2d}. {feat:<35} score={score:.4f}\n")
    f.write("\n")

    # Métricas Tarea 2
    f.write("-" * 60 + "\n")
    f.write("MÉTRICAS TAREA 2 (MMI)\n")
    f.write("-" * 60 + "\n")
    for res in [res_gnb, res_dt]:
        f.write(f"\nModelo  : {res['modelo']}\n")
        f.write(f"  Accuracy  : {res['accuracy']:.4f}  ({res['accuracy']*100:.2f} %)\n")
        f.write(f"  Precision : {res['precision']:.4f}\n")
        f.write(f"  Recall    : {res['recall']:.4f}\n")
        f.write(f"  F1-Score  : {res['f1']:.4f}\n")
        f.write("\n  Matriz de Confusión (filas=Real, columnas=Predicho):\n")
        cm_df = pd.DataFrame(res["cm"], index=CLASS_NAMES, columns=CLASS_NAMES)
        f.write(cm_df.to_string() + "\n")

    # Comparación
    f.write("\n" + "=" * 60 + "\n")
    f.write("COMPARACIÓN TAREA 1 (todas las vars) vs TAREA 2 (MMI)\n")
    f.write("=" * 60 + "\n")
    f.write(f"\n{'Modelo':<35} {'T1 Acc':>8} {'T2 Acc':>8} {'Δ Acc':>8} "
            f"{'T1 F1':>8} {'T2 F1':>8} {'Δ F1':>8}\n")
    f.write("-" * 85 + "\n")

    for key, res in [("Gaussian Naive Bayes", res_gnb),
                     ("Decision Tree (max_depth=6)", res_dt)]:
        t1_acc = T1_METRICS[key]["accuracy"]
        t1_f1  = T1_METRICS[key]["f1"]
        d_acc  = res["accuracy"] - t1_acc
        d_f1   = res["f1"] - t1_f1
        f.write(f"{key:<35} {t1_acc:>8.4f} {res['accuracy']:>8.4f} "
                f"{d_acc:>+8.4f} {t1_f1:>8.4f} {res['f1']:>8.4f} {d_f1:>+8.4f}\n")
    f.write("\n(+) mejora  |  (-) pérdida respecto a Tarea 1\n")

# Imprimir comparación en consola
print("\n" + "=" * 60)
print("COMPARACIÓN TAREA 1 vs TAREA 2")
print("=" * 60)
print(f"\n{'Modelo':<35} {'T1 Acc':>8} {'T2 Acc':>8} {'Δ Acc':>8} "
      f"{'T1 F1':>8} {'T2 F1':>8} {'Δ F1':>8}")
print("-" * 85)
for key, res in [("Gaussian Naive Bayes", res_gnb),
                 ("Decision Tree (max_depth=6)", res_dt)]:
    t1_acc = T1_METRICS[key]["accuracy"]
    t1_f1  = T1_METRICS[key]["f1"]
    d_acc  = res["accuracy"] - t1_acc
    d_f1   = res["f1"] - t1_f1
    print(f"{key:<35} {t1_acc:>8.4f} {res['accuracy']:>8.4f} "
          f"{d_acc:>+8.4f} {t1_f1:>8.4f} {res['f1']:>8.4f} {d_f1:>+8.4f}")
print("\n(+) mejora  |  (-) pérdida respecto a Tarea 1")

print("\n" + "=" * 60)
print("ARCHIVOS GENERADOS")
print(f"  {ranking_path}")
print(f"  {txt_path}")
print(f"  {os.path.join(MODELS_DIR, 'naive_bayes_mmi.pkl')}")
print(f"  {os.path.join(MODELS_DIR, 'decision_tree_mmi.pkl')}")
print("=" * 60)
