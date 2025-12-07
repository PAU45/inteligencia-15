#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LABORATORIO 15 - INTELIGENCIA DE NEGOCIOS
Aplicación de Metodología SEMMA para Análisis de Datos

Integrantes:
- Jairo Jeampiare Quispe Coa
- Paulo Melendez Corrales
- Juan Sanchez Meza
- Patrick Chavez Jimenez
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("APLICACIÓN DE METODOLOGÍA SEMMA EN INTELIGENCIA DE NEGOCIOS")
print("Análisis Predictivo de Comportamiento de Compra de Clientes")
print("=" * 80)

# ============================================================================
# FASE 1: SAMPLE (MUESTREO)
# ============================================================================
print("\n" + "=" * 80)
print("FASE 1: SAMPLE (Muestreo de Datos)")
print("=" * 80)

# Generación de dataset sintético de clientes
np.random.seed(42)
n_total = 10000

print(f"\nGenerando dataset de {n_total} clientes...")

data = {
    'ingreso_cliente': np.random.normal(50000, 15000, n_total),
    'edad': np.random.randint(18, 70, n_total),
    'compras_anteriores': np.random.randint(0, 50, n_total),
    'tiempo_cliente_meses': np.random.randint(1, 120, n_total),
    'descuento_usado': np.random.choice([0, 1], n_total, p=[0.6, 0.4]),
    'satisfaccion': np.random.randint(1, 11, n_total),
    'visitas_web': np.random.randint(0, 100, n_total),
    'tiempo_navegacion_min': np.random.randint(1, 180, n_total)
}

# Crear variable objetivo basada en reglas de negocio
data['compra'] = (
    (data['ingreso_cliente'] > 45000) & 
    (data['satisfaccion'] > 5) & 
    (data['compras_anteriores'] > 10) &
    (data['visitas_web'] > 20)
).astype(int)

df = pd.DataFrame(data)

# Tomar muestra representativa (30% del total)
sample_size = int(len(df) * 0.3)
df_sample = df.sample(n=sample_size, random_state=42)

print(f"\n✓ Dataset original: {len(df):,} registros")
print(f"✓ Muestra tomada: {len(df_sample):,} registros ({sample_size/len(df)*100:.1f}%)")
print(f"\nPrimeros registros de la muestra:")
print(df_sample.head(10))

print(f"\nInformación del dataset:")
print(df_sample.info())

# ============================================================================
# FASE 2: EXPLORE (EXPLORACIÓN)
# ============================================================================
print("\n" + "=" * 80)
print("FASE 2: EXPLORE (Exploración de Datos)")
print("=" * 80)

# Estadísticas descriptivas
print("\nEstadísticas Descriptivas:")
print(df_sample.describe())

# Distribución de variable objetivo
print("\nDistribución de la Variable Objetivo (Compra):")
compra_dist = df_sample['compra'].value_counts()
print(compra_dist)
print(f"\nPorcentaje de conversión: {compra_dist[1]/len(df_sample)*100:.2f}%")

# Análisis de correlación
print("\nMatriz de Correlación con Variable Objetivo:")
correlacion = df_sample.corr()
print(correlacion['compra'].sort_values(ascending=False))

# Visualizaciones exploratorias
print("\nGenerando visualizaciones exploratorias...")

fig = plt.figure(figsize=(16, 12))

# Gráfico 1: Distribución de Compras
plt.subplot(3, 3, 1)
compra_counts = df_sample['compra'].value_counts()
bars = plt.bar(['No Compra', 'Compra'], compra_counts.values, color=['#e74c3c', '#2ecc71'])
plt.title('Distribución de Compras', fontsize=12, fontweight='bold')
plt.ylabel('Cantidad de Clientes')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')

# Gráfico 2: Ingreso vs Satisfacción
plt.subplot(3, 3, 2)
scatter = plt.scatter(df_sample['ingreso_cliente'], 
                     df_sample['satisfaccion'],
                     c=df_sample['compra'], 
                     cmap='RdYlGn', 
                     alpha=0.6,
                     edgecolors='black',
                     linewidth=0.5)
plt.colorbar(scatter, label='Compra')
plt.xlabel('Ingreso Cliente ($)')
plt.ylabel('Satisfacción (1-10)')
plt.title('Ingreso vs Satisfacción por Compra', fontsize=12, fontweight='bold')

# Gráfico 3: Distribución de Edad
plt.subplot(3, 3, 3)
df_sample.boxplot(column='edad', by='compra', ax=plt.gca())
plt.title('Distribución de Edad por Compra', fontsize=12, fontweight='bold')
plt.suptitle('')
plt.xlabel('Compra (0=No, 1=Sí)')
plt.ylabel('Edad (años)')

# Gráfico 4: Mapa de Calor de Correlaciones
plt.subplot(3, 3, 4)
sns.heatmap(correlacion, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Matriz de Correlación', fontsize=12, fontweight='bold')

# Gráfico 5: Ingreso por Compra
plt.subplot(3, 3, 5)
df_sample.boxplot(column='ingreso_cliente', by='compra', ax=plt.gca())
plt.title('Ingreso por Tipo de Compra', fontsize=12, fontweight='bold')
plt.suptitle('')
plt.xlabel('Compra (0=No, 1=Sí)')
plt.ylabel('Ingreso ($)')

# Gráfico 6: Compras Anteriores
plt.subplot(3, 3, 6)
df_sample.boxplot(column='compras_anteriores', by='compra', ax=plt.gca())
plt.title('Compras Anteriores', fontsize=12, fontweight='bold')
plt.suptitle('')
plt.xlabel('Compra (0=No, 1=Sí)')
plt.ylabel('N° Compras Anteriores')

# Gráfico 7: Histograma de Ingresos
plt.subplot(3, 3, 7)
df_sample[df_sample['compra']==0]['ingreso_cliente'].hist(alpha=0.7, bins=30, label='No Compra', color='red')
df_sample[df_sample['compra']==1]['ingreso_cliente'].hist(alpha=0.7, bins=30, label='Compra', color='green')
plt.xlabel('Ingreso Cliente ($)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Ingresos', fontsize=12, fontweight='bold')
plt.legend()

# Gráfico 8: Visitas Web
plt.subplot(3, 3, 8)
df_sample.boxplot(column='visitas_web', by='compra', ax=plt.gca())
plt.title('Visitas Web por Compra', fontsize=12, fontweight='bold')
plt.suptitle('')
plt.xlabel('Compra (0=No, 1=Sí)')
plt.ylabel('N° Visitas Web')

# Gráfico 9: Satisfacción
plt.subplot(3, 3, 9)
satisfaccion_compra = df_sample.groupby(['satisfaccion', 'compra']).size().unstack(fill_value=0)
satisfaccion_compra.plot(kind='bar', stacked=True, color=['#e74c3c', '#2ecc71'], ax=plt.gca())
plt.xlabel('Satisfacción')
plt.ylabel('Cantidad')
plt.title('Satisfacción vs Compra', fontsize=12, fontweight='bold')
plt.legend(['No Compra', 'Compra'])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('fase_explore.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: fase_explore.png")
plt.close()

# ============================================================================
# FASE 3: MODIFY (MODIFICACIÓN Y PREPARACIÓN)
# ============================================================================
print("\n" + "=" * 80)
print("FASE 3: MODIFY (Modificación y Preparación de Datos)")
print("=" * 80)

df_clean = df_sample.copy()

# 1. Detección y tratamiento de valores atípicos
print("\nTratamiento de valores atípicos...")
for col in ['ingreso_cliente', 'edad', 'compras_anteriores']:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    outliers_antes = len(df_clean)
    df_clean = df_clean[
        (df_clean[col] >= limite_inferior) & 
        (df_clean[col] <= limite_superior)
    ]
    outliers_removidos = outliers_antes - len(df_clean)
    print(f"  - {col}: {outliers_removidos} outliers removidos")

print(f"\n✓ Registros después de limpieza: {len(df_clean)}")

# 2. Creación de nuevas variables (Feature Engineering)
print("\nCreando nuevas variables (Feature Engineering)...")

df_clean['ingreso_edad_ratio'] = df_clean['ingreso_cliente'] / df_clean['edad']
df_clean['compras_por_mes'] = df_clean['compras_anteriores'] / (df_clean['tiempo_cliente_meses'] / 12)
df_clean['engagement_score'] = (df_clean['visitas_web'] * df_clean['tiempo_navegacion_min']) / 100
df_clean['cliente_premium'] = ((df_clean['ingreso_cliente'] > 60000) & 
                               (df_clean['compras_anteriores'] > 20)).astype(int)

print("✓ Variables creadas:")
print("  - ingreso_edad_ratio")
print("  - compras_por_mes")
print("  - engagement_score")
print("  - cliente_premium")

# 3. Normalización de variables numéricas
print("\nNormalizando variables numéricas...")

scaler = StandardScaler()
columnas_numericas = ['ingreso_cliente', 'edad', 'compras_anteriores', 
                      'tiempo_cliente_meses', 'satisfaccion', 'visitas_web',
                      'tiempo_navegacion_min', 'ingreso_edad_ratio', 
                      'compras_por_mes', 'engagement_score']

df_clean[columnas_numericas] = scaler.fit_transform(df_clean[columnas_numericas])

print("✓ Variables normalizadas con StandardScaler")
print("\nPrimeros registros después de la modificación:")
print(df_clean.head())

# ============================================================================
# FASE 4: MODEL (MODELADO)
# ============================================================================
print("\n" + "=" * 80)
print("FASE 4: MODEL (Modelado Predictivo)")
print("=" * 80)

# Preparar datos para modelado
X = df_clean.drop('compra', axis=1)
y = df_clean['compra']

print(f"\nShape de características (X): {X.shape}")
print(f"Shape de variable objetivo (y): {y.shape}")

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
)

print(f"\n✓ Datos de entrenamiento: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Datos de prueba: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

print(f"\nDistribución en entrenamiento:")
print(f"  - No Compra: {(y_train==0).sum()}")
print(f"  - Compra: {(y_train==1).sum()}")

# Entrenar modelo Random Forest
print("\nEntrenando modelo Random Forest...")

modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

modelo.fit(X_train, y_train)
print("✓ Modelo Random Forest entrenado exitosamente")

# Importancia de variables
importancias = pd.DataFrame({
    'variable': X.columns,
    'importancia': modelo.feature_importances_
}).sort_values('importancia', ascending=False)

print("\nImportancia de Variables (Top 10):")
print(importancias.head(10).to_string(index=False))

# Visualizar importancia de variables
plt.figure(figsize=(12, 8))
top_n = 10
top_vars = importancias.head(top_n)
bars = plt.barh(range(len(top_vars)), top_vars['importancia'], color='steelblue')
plt.yticks(range(len(top_vars)), top_vars['variable'])
plt.xlabel('Importancia', fontsize=12)
plt.title('Top 10 Variables Más Importantes para Predicción de Compra', 
          fontsize=14, fontweight='bold', pad=20)
plt.gca().invert_yaxis()

# Agregar valores en las barras
for i, (idx, row) in enumerate(top_vars.iterrows()):
    plt.text(row['importancia'], i, f' {row["importancia"]:.4f}', 
             va='center', fontsize=10)

plt.tight_layout()
plt.savefig('importancia_variables.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: importancia_variables.png")
plt.close()

# ============================================================================
# FASE 5: ASSESS (EVALUACIÓN)
# ============================================================================
print("\n" + "=" * 80)
print("FASE 5: ASSESS (Evaluación del Modelo)")
print("=" * 80)

# Predicciones
y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)

# Métricas de evaluación
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("\n" + "-" * 80)
print("MÉTRICAS DE RENDIMIENTO")
print("-" * 80)
print(f"\nPrecisión en Entrenamiento: {accuracy_train:.4f} ({accuracy_train*100:.2f}%)")
print(f"Precisión en Prueba:        {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
print(f"Diferencia (Overfitting):   {abs(accuracy_train - accuracy_test):.4f}")

if abs(accuracy_train - accuracy_test) < 0.05:
    print("✓ El modelo NO presenta overfitting significativo")
else:
    print("⚠ El modelo podría presentar overfitting")

# Reporte de clasificación detallado
print("\n" + "-" * 80)
print("REPORTE DE CLASIFICACIÓN (Conjunto de Prueba)")
print("-" * 80)
print(classification_report(y_test, y_pred_test, 
                          target_names=['No Compra', 'Compra'],
                          digits=4))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_test)

print("\n" + "-" * 80)
print("MATRIZ DE CONFUSIÓN")
print("-" * 80)
print(f"\nVerdaderos Negativos (TN): {cm[0,0]}")
print(f"Falsos Positivos (FP):     {cm[0,1]}")
print(f"Falsos Negativos (FN):     {cm[1,0]}")
print(f"Verdaderos Positivos (TP): {cm[1,1]}")

# Métricas adicionales
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print(f"\nSensibilidad (Recall):     {sensitivity:.4f}")
print(f"Especificidad:             {specificity:.4f}")

# Visualizar matriz de confusión
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Matriz de confusión - Valores absolutos
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Compra', 'Compra'],
            yticklabels=['No Compra', 'Compra'],
            cbar_kws={'label': 'Cantidad'})
axes[0].set_title('Matriz de Confusión\n(Valores Absolutos)', 
                  fontsize=14, fontweight='bold', pad=15)
axes[0].set_ylabel('Valor Real', fontsize=12)
axes[0].set_xlabel('Valor Predicho', fontsize=12)

# Matriz de confusión - Porcentajes
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Greens', ax=axes[1],
            xticklabels=['No Compra', 'Compra'],
            yticklabels=['No Compra', 'Compra'],
            cbar_kws={'label': 'Porcentaje (%)'})
axes[1].set_title('Matriz de Confusión\n(Porcentajes)', 
                  fontsize=14, fontweight='bold', pad=15)
axes[1].set_ylabel('Valor Real', fontsize=12)
axes[1].set_xlabel('Valor Predicho', fontsize=12)

plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: matriz_confusion.png")
plt.close()

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN DEL ANÁLISIS SEMMA")
print("=" * 80)

print("\n✓ FASE 1 - SAMPLE: Muestreo de 3,000 registros de 10,000 totales")
print(f"✓ FASE 2 - EXPLORE: Análisis exploratorio completado")
print(f"✓ FASE 3 - MODIFY: {len(df_clean)} registros limpios, 4 nuevas variables creadas")
print(f"✓ FASE 4 - MODEL: Random Forest con {modelo.n_estimators} árboles entrenado")
print(f"✓ FASE 5 - ASSESS: Precisión de {accuracy_test*100:.2f}% en conjunto de prueba")

print("\n" + "=" * 80)
print("ARCHIVOS GENERADOS")
print("=" * 80)
print("  1. fase_explore.png           - Visualizaciones exploratorias (9 gráficos)")
print("  2. importancia_variables.png  - Top 10 variables más importantes")
print("  3. matriz_confusion.png       - Matriz de confusión (absoluta y porcentual)")

print("\n" + "=" * 80)
print("ANÁLISIS SEMMA COMPLETADO EXITOSAMENTE")
print("=" * 80)
print("\nLos resultados demuestran la efectividad de la metodología SEMMA")
print("para proyectos de inteligencia de negocios y análisis predictivo.")
print("\n")