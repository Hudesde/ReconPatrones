"""
Visualización — Ranking de Información Mutua (MMI)
Gráfica 1: Target = anxiety_level
Gráfica 2: Target = depression

Diseñado para ejecutarse de forma independiente o pegarse al final de tarea2_mmi.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler

# =============================================================================
# RUTAS
# =============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_CSV    = os.path.join(BASE_DIR, "StressLevelDataset.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# CARGA Y PREPROCESAMIENTO LIVIANO
# =============================================================================
df = pd.read_csv(DATA_CSV)

TARGETS = {
    "anxiety_level": {
        "titulo": "Ranking de Características por Información Mutua\nTarget: Nivel de Ansiedad",
        "color":  "#4472C4",          # azul corporativo
        "archivo": "ranking_mmi_anxiety.png",
    },
    "depression": {
        "titulo": "Ranking de Características por Información Mutua\nTarget: Nivel de Depresión",
        "color":  "#E85D49",          # coral / rojo suave
        "archivo": "ranking_mmi_depression.png",
    },
}

# =============================================================================
# FUNCIÓN PRINCIPAL DE GRAFICACIÓN
# =============================================================================
def graficar_mmi(target_col: str, cfg: dict) -> pd.DataFrame:
    """
    Calcula el score de Información Mutua para cada variable
    respecto a `target_col`, genera la gráfica y la guarda en /results.
    Devuelve el DataFrame del ranking.
    """
    # Excluir la columna objetivo y stress_level del conjunto de features
    excluir = {target_col, "stress_level"}
    feature_cols = [c for c in df.columns if c not in excluir]

    X = df[feature_cols].values
    y = df[target_col].values

    # Escalar para coherencia con el pipeline principal
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # mutual_info_classif es válido para targets discretos (int)
    mi_scores = mutual_info_classif(X_scaled, y, random_state=42)

    ranking = (
        pd.DataFrame({"caracteristica": feature_cols, "mi_score": mi_scores})
        .sort_values("mi_score", ascending=True)   # ascending=True → barh de menor a mayor (visualmente de arriba abajo queda mayor arriba)
        .reset_index(drop=True)
    )

    # ── Gráfica ──────────────────────────────────────────────────────────────
    n = len(ranking)
    fig_height = max(6, n * 0.45)

    fig, ax = plt.subplots(figsize=(10, fig_height), facecolor="white")
    ax.set_facecolor("white")

    bars = ax.barh(
        ranking["caracteristica"],
        ranking["mi_score"],
        color=cfg["color"],
        edgecolor="white",
        linewidth=0.6,
        height=0.65,
    )

    # Etiqueta de valor al final de cada barra
    for bar, val in zip(bars, ranking["mi_score"]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left",
            fontsize=8.5, color="#333333",
        )

    # Títulos y ejes
    ax.set_title(cfg["titulo"], fontsize=13, fontweight="bold",
                 pad=14, color="#1a1a1a")
    ax.set_xlabel("Información Mutua (MI Score)", fontsize=10, labelpad=8)
    ax.set_ylabel("Características", fontsize=10, labelpad=8)

    # Límite X con margen para las etiquetas
    ax.set_xlim(0, ranking["mi_score"].max() * 1.18)

    # Rejilla vertical sutil
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="x", linestyle="--", linewidth=0.5,
            alpha=0.6, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    # Quitar bordes superior y derecho
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#aaaaaa")
    ax.spines["bottom"].set_color("#aaaaaa")

    # Tamaño de fuente en eje Y
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8.5)

    plt.tight_layout(pad=1.5)

    out_path = os.path.join(RESULTS_DIR, cfg["archivo"])
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    print(f"  ✔  Guardada: {out_path}")
    return ranking

# =============================================================================
# GENERAR AMBAS GRÁFICAS
# =============================================================================
print("=" * 60)
print("GENERANDO GRÁFICAS DE RANKING MMI")
print("=" * 60)

for target, config in TARGETS.items():
    print(f"\n  Target: {target}")
    ranking_df = graficar_mmi(target, config)
    print(f"     Top-3: {ranking_df['caracteristica'].iloc[-1]}  "
          f"({ranking_df['mi_score'].iloc[-1]:.4f})  |  "
          f"{ranking_df['caracteristica'].iloc[-2]}  "
          f"({ranking_df['mi_score'].iloc[-2]:.4f})  |  "
          f"{ranking_df['caracteristica'].iloc[-3]}  "
          f"({ranking_df['mi_score'].iloc[-3]:.4f})")

print("\n" + "=" * 60)
print("Gráficas listas en /results/")
print("=" * 60)
