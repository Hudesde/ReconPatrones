"""
Fisher Score (F-score / ANOVA F-value) — Ranking de características
Comparación con el ranking de Información Mutua (MMI)
"""

import os
import pandas as pd
from sklearn.feature_selection import f_classif

# =============================================================================
# RUTAS
# =============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data_processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Top-5 MMI de referencia (obtenido en Tarea 2)
TOP5_MMI = ["blood_pressure", "future_career_concerns",
            "sleep_quality", "bullying", "self_esteem"]

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================
X_train_df = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train    = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()

feature_names = X_train_df.columns.tolist()
X_train       = X_train_df.values

# =============================================================================
# 2. CÁLCULO DEL FISHER SCORE (ANOVA F-value)
#    f_classif devuelve (F-statistic, p-value) para cada variable.
#    Un F-value alto indica mayor separación entre clases.
# =============================================================================
f_values, p_values = f_classif(X_train, y_train)

ranking_fisher = pd.DataFrame({
    "caracteristica": feature_names,
    "f_score":        f_values,
    "p_value":        p_values,
}).sort_values("f_score", ascending=False).reset_index(drop=True)

ranking_fisher.index += 1   # posición empieza en 1

# =============================================================================
# 3. GUARDAR RANKING
# =============================================================================
out_path = os.path.join(RESULTS_DIR, "ranking_fisher.csv")
ranking_fisher.to_csv(out_path)
print("=" * 65)
print("RANKING COMPLETO — FISHER SCORE (ANOVA F-value)")
print("=" * 65)
print(ranking_fisher.to_string(float_format=lambda x: f"{x:.4f}"))
print(f"\nGuardado en: {out_path}")

# =============================================================================
# 4. COMPARACIÓN TOP-5 FISHER vs TOP-5 MMI
# =============================================================================
top5_fisher = ranking_fisher["caracteristica"].head(5).tolist()

print("\n" + "=" * 65)
print("COMPARACIÓN TOP-5: FISHER vs MMI")
print("=" * 65)
print(f"\n{'Pos':>4}  {'Fisher Score':^35}  {'MMI':^35}  {'¿Coincide?':^10}")
print("-" * 90)

todas_coinciden = True
for i, (f_feat, m_feat) in enumerate(zip(top5_fisher, TOP5_MMI), 1):
    coincide = "✔" if f_feat == m_feat else "✘"
    if f_feat != m_feat:
        todas_coinciden = False
    print(f"  {i:>2}  {f_feat:<35}  {m_feat:<35}  {coincide:^10}")

print()
if todas_coinciden:
    print("  ✔ El Top-5 de Fisher y el de MMI son IDÉNTICOS.")
else:
    # Calcular discrepancias
    solo_fisher = [f for f in top5_fisher if f not in TOP5_MMI]
    solo_mmi    = [m for m in TOP5_MMI    if m not in top5_fisher]

    print("  ✘ Existen discrepancias entre los rankings:\n")
    if solo_fisher:
        print(f"  Variables en Top-5 Fisher pero NO en Top-5 MMI : {solo_fisher}")
    if solo_mmi:
        print(f"  Variables en Top-5 MMI    pero NO en Top-5 Fisher: {solo_mmi}")

    # Mostrar posición de cada variable discrepante en ambos rankings
    print("\n  Detalle de posiciones para variables discrepantes:")
    discrepantes = set(solo_fisher + solo_mmi)
    mmi_ranking  = pd.read_csv(os.path.join(RESULTS_DIR, "ranking_mmi.csv"),
                               index_col=0)
    mmi_ranking.index = range(1, len(mmi_ranking) + 1)

    print(f"\n  {'Variable':<35}  {'Pos Fisher':>10}  {'Pos MMI':>8}")
    print("  " + "-" * 57)
    for var in sorted(discrepantes):
        pos_f = ranking_fisher[ranking_fisher["caracteristica"] == var].index
        pos_m = mmi_ranking[mmi_ranking["caracteristica"] == var].index
        pf = pos_f[0] if len(pos_f) else "—"
        pm = pos_m[0] if len(pos_m) else "—"
        print(f"  {var:<35}  {str(pf):>10}  {str(pm):>8}")

print("\n" + "=" * 65)
print("RESUMEN TOP-10 LADO A LADO")
print("=" * 65)
mmi_ranking = pd.read_csv(os.path.join(RESULTS_DIR, "ranking_mmi.csv"),
                          index_col=0)
mmi_ranking.index = range(1, len(mmi_ranking) + 1)

print(f"\n{'Pos':>4}  {'Fisher':<35}  {'F-score':>10}  "
      f"{'MMI':<35}  {'MI-score':>10}")
print("-" * 100)
for i in range(1, 11):
    f_row = ranking_fisher.loc[i]
    m_row = mmi_ranking.loc[i]
    mark = "  " if f_row["caracteristica"] == m_row["caracteristica"] else "◄"
    print(f"  {i:>2}  {f_row['caracteristica']:<35}  "
          f"{f_row['f_score']:>10.4f}  "
          f"{m_row['caracteristica']:<35}  "
          f"{m_row['mi_score']:>10.4f}  {mark}")

print("\n(◄ indica posición diferente entre los dos rankings)")
