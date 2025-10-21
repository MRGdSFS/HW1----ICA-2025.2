import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import funcoes_analise_uni as au


# Função destinada a separar as variáveis em qualitativas e quantitativas, permitindo a análise bivariada adequada.
def separar_variaveis(df_dict: pd.DataFrame) -> tuple:
    quantitativas, qualitativas = au.separar_variaveis(df_dict)
    quantitativas.append('exam_score')
    qualitativas.append('performance_class')
    return quantitativas, qualitativas

# Função para comparar variáveis quantitativas com scatter plots coloridos por 'performance_class'.
def comparar_quantitativas(df, quantitativas: list, figuras: str) -> None:
    pares_quant = [(quantitativas[i], quantitativas[j]) 
                   for i in range(len(quantitativas)) 
                   for j in range(i+1, len(quantitativas))]
    
    n_cols = 3
    n_rows = math.ceil(len(pares_quant) / n_cols)
    
    if pares_quant:
        fig, subs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        subs = subs.flatten()
        
        for idx, (x_col, y_col) in enumerate(pares_quant):
            sns.scatterplot(
                data=df,
                x=x_col, y=y_col,
                hue='performance_class',  # Colorir pelas classes!
                palette='viridis',
                ax=subs[idx],
                alpha=0.7, edgecolor='none'
            )
            subs[idx].set_title(f'{x_col} vs {y_col}')
            subs[idx].set_xlabel(x_col)
            subs[idx].set_ylabel(y_col)
            subs[idx].legend().set_title('Performance')
        
        for idx in range(len(pares_quant), n_rows * n_cols):
            fig.delaxes(subs[idx])
        
        plt.suptitle('Bivariada Incondicional - Quantitativa x Quantitativa (Colorido por performance_class)', fontsize=15, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{figuras}/bivariada_quantxquant_colorida.png", bbox_inches='tight')
        plt.show()

# Função para gerar um heatmap de correlação entre variáveis quantitativas.
def heatmap_quantitativas(df, quantitativas: list, figuras: str) -> None:
    corr = df[quantitativas].corr(method='pearson')
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlação entre Variáveis Quantitativas (Heatmap Global)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{figuras}/heatmap_global_quantxquant.png", bbox_inches='tight')
    plt.show()

# Função para comparar variáveis qualitativas com heatmaps de tabelas cruzadas.
def comparar_qualitativas(df, qualitativas: list, figuras: str) -> None:
    # Cria todos os pares possíveis
    pares_qual = [(qualitativas[i], qualitativas[j]) 
                  for i in range(len(qualitativas)) 
                  for j in range(i+1, len(qualitativas))]

    # Separa pares com performance_class e sem
    pares_com_perf = [p for p in pares_qual if 'performance_class' in p]
    pares_sem_perf = [p for p in pares_qual if 'performance_class' not in p]

    n_cols = 3

    n_rows = math.ceil(len(pares_sem_perf) / n_cols)
    fig, subs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    subs = subs.flatten()
        
    for idx, (col1, col2) in enumerate(pares_sem_perf):
        crosstab = pd.crosstab(df[col1], df[col2])
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd', ax=subs[idx])
        subs[idx].set_title(f'{col1} vs {col2}')
        
    # Remove subplots extras
    for idx in range(len(pares_sem_perf), n_rows * n_cols):
        fig.delaxes(subs[idx])
        
    plt.suptitle('Bivariada - Qualitativa x Qualitativa', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{figuras}/{"bivariada_qualxqual_sem_perf"}.png", bbox_inches='tight')
    plt.show()

    # Função auxiliar para plotar histogramas empilhados de performance_class
    n_rows = math.ceil(len(pares_com_perf) / n_cols)
    fig, subs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    subs = subs.flatten()

    for idx, (col1, col2) in enumerate(pares_com_perf):
        # Determina a variável qualitativa que não é performance_class
        qual_col = col1 if col2 == 'performance_class' else col2
        prop_df = (
            df.groupby([qual_col, 'performance_class'])
                .size()
              .groupby(level=0)
              .apply(lambda x: x / x.sum())
              .unstack(fill_value=0)
        )
        prop_df.plot(
            kind='bar',
            stacked=True,
            colormap='viridis',
            ax=subs[idx]
        )
        subs[idx].set_title(f'{qual_col} vs performance_class')
        subs[idx].set_xlabel(qual_col)
        subs[idx].set_ylabel('Proporção')
        subs[idx].legend(title='Performance Class', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Remove subplots extras
    for idx in range(len(pares_com_perf), n_rows * n_cols):
        fig.delaxes(subs[idx])

    plt.suptitle('Qualitativa x performance_class', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{figuras}/{"qualitativa_vs_performance"}.png", bbox_inches='tight')
    plt.show()
