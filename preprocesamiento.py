import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =============================================================================
# CREACIÓN DE LA ESTRUCTURA DE CARPETAS DEL PROYECTO
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIRS = ["data_processed", "models", "results"]

for folder in DIRS:
    path = os.path.join(BASE_DIR, folder)
    os.makedirs(path, exist_ok=True)

print("Estructura de carpetas creada: /data_processed  /models  /results")


# =============================================================================
# 1. CARGA Y LIMPIEZA
# =============================================================================
df = pd.read_csv("StressLevelDataset.csv")

print("=" * 50)
print("INFORMACIÓN GENERAL DEL DATASET")
print("=" * 50)
print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nTipos de datos:\n{df.dtypes}")

# Verificación de valores nulos
nulos = df.isnull().sum()
print(f"\nValores nulos por columna:\n{nulos}")
print(f"\nTotal de valores nulos: {nulos.sum()}")

# Verificación de duplicados
duplicados = df.duplicated().sum()
print(f"\nFilas duplicadas: {duplicados}")

if duplicados > 0:
    df = df.drop_duplicates()
    print(f"  -> Duplicados eliminados. Nuevo tamaño: {df.shape}")

# =============================================================================
# 2. ANÁLISIS DE LA VARIABLE OBJETIVO
# =============================================================================
TARGET_COL = "stress_level"

print("\n" + "=" * 50)
print("DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
print("=" * 50)
freq = df[TARGET_COL].value_counts().sort_index()
freq_pct = df[TARGET_COL].value_counts(normalize=True).sort_index() * 100

distribucion = pd.DataFrame({"Frecuencia": freq, "Porcentaje (%)": freq_pct.round(2)})
print(distribucion)

clases_balanceadas = freq_pct.max() - freq_pct.min() < 10
print(f"\n¿Clases aproximadamente balanceadas? {'Sí' if clases_balanceadas else 'No'}")

# =============================================================================
# 3. SEPARACIÓN DE CARACTERÍSTICAS
# =============================================================================
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print("\n" + "=" * 50)
print("SEPARACIÓN DE CARACTERÍSTICAS")
print("=" * 50)
print(f"Variables independientes (X): {X.shape[1]} columnas")
print(f"  {list(X.columns)}")
print(f"Variable objetivo (y): '{TARGET_COL}'  |  Clases: {sorted(y.unique())}")

# =============================================================================
# 4. ESCALADO DE DATOS (StandardScaler)
# =============================================================================
scaler = StandardScaler()

# Nota: el scaler se ajusta SOLO sobre el conjunto de entrenamiento.
# Aquí hacemos la división primero y luego escalamos para evitar data leakage.

# =============================================================================
# 5. DIVISIÓN HOLD-OUT  (70% train / 30% test)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y        # mantiene la proporción de clases en ambos subconjuntos
)

# Ajuste del scaler con datos de entrenamiento y transformación de ambos conjuntos
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("\n" + "=" * 50)
print("DIVISIÓN HOLD-OUT  (70 / 30)")
print("=" * 50)
print(f"X_train: {X_train.shape}  |  y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}   |  y_test : {y_test.shape}")

print("\n" + "=" * 50)
print("DIVISIÓN HOLD-OUT  (70 / 30)")
print("=" * 50)
print(f"X_train: {X_train.shape}  |  y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}   |  y_test : {y_test.shape}")

# =============================================================================
# 6. GUARDAR DATOS PROCESADOS EN /data_processed
# =============================================================================
DATA_DIR = os.path.join(BASE_DIR, "data_processed")

# Guardar como CSV (las columnas originales se pierden tras StandardScaler,
# se usan nombres genéricos feat_0, feat_1, ...)
feature_names = X.columns.tolist()

pd.DataFrame(X_train, columns=feature_names).to_csv(
    os.path.join(DATA_DIR, "X_train.csv"), index=False)
pd.DataFrame(X_test,  columns=feature_names).to_csv(
    os.path.join(DATA_DIR, "X_test.csv"),  index=False)
y_train.reset_index(drop=True).to_csv(
    os.path.join(DATA_DIR, "y_train.csv"), index=False, header=["stress_level"])
y_test.reset_index(drop=True).to_csv(
    os.path.join(DATA_DIR, "y_test.csv"),  index=False, header=["stress_level"])

print("\n" + "=" * 50)
print("PREPROCESAMIENTO COMPLETADO")
print(f"Datos guardados en: {DATA_DIR}")
print("  X_train.csv  X_test.csv  y_train.csv  y_test.csv")
print("=" * 50)
