#funções reutilizáveis para análise bivariada incondicional

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregamento dos dados do .csv a partir do caminho
def carregar_dados(path):
    df = pd.read_csv(path)
    return df

def analise_bivariada(df: pd.DataFrame, classe_col: str, figuras: str, tables: str) -> pd.DataFrame:
    os.makedirs(figuras, exist_ok=True)

    # Seleciona apenas as variáveis numéricas (preditoras)
    numericas = df.select_dtypes(include=[np.number])

    # =====================================================
    # 1️⃣ Geração de scatter plots coloridos por classe
    # =====================================================
    colunas = numericas.columns

    for i in range(len(colunas)):
        for j in range(i + 1, len(colunas)):  # evita pares repetidos e diagonais
            x_col = colunas[i]
            y_col = colunas[j]

            plt.figure(figsize=(6, 4))
            sns.scatterplot(
                data=df,
                x=x_col,
                y=y_col,
                hue=classe_col,         # cor conforme a classe
                palette='viridis',
                alpha=0.7,
                edgecolor='none'
            )
            plt.title(f'{x_col} vs {y_col} por {classe_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.legend(title=classe_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            plt.savefig(f"{figuras}/scatter_{x_col}_vs_{y_col}.png", bbox_inches='tight')
            plt.close()

    # =====================================================
    # 2️⃣ Cálculo e visualização da matriz de correlação
    # =====================================================
    corr_matrix = numericas.corr()

    # salva em CSV para registro
    os.makedirs(tables, exist_ok=True)
    corr_matrix.to_csv(f"{tables}/matriz_correlacao.csv")

    # plota como heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt=".2f",
        linewidths=0.5
    )
    plt.title('Matriz de Correlação entre Preditores Numéricos')
    plt.tight_layout()

    plt.savefig(f"{figuras}/heatmap_correlacao.png", bbox_inches='tight')
    plt.close()

    print("✅ Análise bivariada concluída! Scatter plots e correlação salvos.")
    return corr_matrix
