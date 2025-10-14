#funções reutilizáveis para análise univariada incondicional e condicionada por classe

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

#função pra carregar o dataset e retornar um df (path é o caminho do csv salvo no computador)
def carregar_dados(path):
    df = pd.read_csv(path)
    return df

#função para termos certezas que as pastas de figuras e tables existem e podemos salvar nossos resultados lá
def garantir_pastas():
    os.makedirs("notebooks/resultados/figuras", exist_ok=True)
    os.makedirs("notebooks/resultados/tables", exist_ok=True)

#função pra calcular a média, desvio padrão e assimetria
#usamos a função select_dtypes para filtrar as colunas numéricas automaticamente do dataframe
def estatisticas(df):
    numero_colunas = df.select_dtypes(include=[np.number])
    sumario = pd.DataFrame({
        'media': numero_colunas.mean(),
        'desvio_padrao': numero_colunas.std(),
        'assimetria': numero_colunas.apply(skew)
    })
    return sumario

#função para plotar os histogramas das variáveis numéricas, aqui usamos a função da biblioteca seaborn
def plotar_histogramas(df, figuras="notebooks/resultados/figuras"):
    numero_colunas = df.select_dtypes(include=[np.number])
    for coluna in numero_colunas.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[coluna], kde=True, color='skyblue')
        plt.title(f'Histograma de {coluna}')
        plt.xlabel(coluna)
        plt.ylabel('Frequência')
        plt.tight_layout()
         
        plt.savefig(f"{figuras}/hist_{coluna}.png", bbox_inches='tight')
        plt.close()

#função pra plotar os boxplots das variáveis numéricas, aqui usamos a função da biblioteca seaborn
def plotar_boxplots(df, figuras="notebooks/resultados/figuras"):
    numero_colunas = df.select_dtypes(include=[np.number])
    for coluna in numero_colunas.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[coluna], color='lightcoral')
        plt.title(f'Boxplot de {coluna}')
        plt.xlabel(coluna)
        plt.tight_layout()
         
        plt.savefig(f"{figuras}/box_{coluna}.png", bbox_inches='tight')
        plt.close()

#função que vai discretizar a variável categórica exam_score, que é o nosso target, em 4 grupos
def categorizar_performance(df):
    conditions = [
        (df['exam_score'] < 40),
        (df['exam_score'] >= 40) & (df['exam_score'] < 70),
        (df['exam_score'] >= 70) & (df['exam_score'] < 90),
        (df['exam_score'] >= 90)
    ]
    choices = ['Reprovado', 'Recuperação', 'Bom', 'Excelente']
    df['Performance'] = np.select(conditions, choices, default="Indefinido")
    return df

df = carregar_dados("/home/natan/Área de trabalho/HW1-ICA/dados/student_habits_performance.csv")

#a partir daqui a variável exam_score está dividida em 4 grupos e não possui uma barra pra cada valor como as outras variáveis categóricas
df = categorizar_performance(df)

#função pra gerar os gráficos em barras das variáveis não numéricas, usamos a função value_counts() para pegar os valores únicos e deixarem eles em suas próprias barras no gráfico
def analise_categoricas(df, figuras="notebooks/resultados/figuras"):
    categorias = df.select_dtypes(exclude=[np.number])
    for coluna in categorias.columns:
        plt.figure(figsize=(6, 4))
        df[coluna].value_counts().plot(kind='bar', color='lightgreen')
        plt.title(f'Distribuição de {coluna}')
        plt.xlabel(coluna)
        plt.ylabel('Frequência')
        plt.tight_layout()
         
        plt.savefig(f"{figuras}/cat_{coluna}.png", bbox_inches='tight')
        plt.close()

#função que faz a análise univariada condicionada por classe, que também calcula a média, desvio padrão e assimetria gerando box-plots e histogramas para as variáveis númericas
#e para as categóricas gera gráficos de barras mostrando a distribuição por categoria 
def analise_univariada_condicional(df, classe_col="Performance", figuras="notebooks/resultados/figuras"):
    os.makedirs(figuras, exist_ok=True)
    #separa variáveis numéricas e categóricas
    numericas = df.select_dtypes(include=[np.number])
    categoricas = df.select_dtypes(exclude=[np.number])
    classes = df[classe_col].unique()

    #numéricas
    resultados = []

    for classe in classes:
        subset = df[df[classe_col] == classe]
        estat = pd.DataFrame({
            'media': subset[numericas.columns].mean(),
            'desvio_padrao': subset[numericas.columns].std(),
            'assimetria': subset[numericas.columns].apply(skew)
        })
        estat['classe'] = classe
        resultados.append(estat)

        for col in numericas.columns:
            #histograma por classe
            plt.figure(figsize=(6, 4))
            sns.histplot(subset[col], kde=True, color='steelblue')
            plt.title(f'{col} - Classe: {classe}')
            plt.xlabel(col)
            plt.ylabel('Frequência')
            plt.tight_layout()
             
            plt.savefig(f"{figuras}/hist_{col}_{classe}.png", bbox_inches='tight')
            plt.close()

            #boxplot por classe
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=subset[col], color='salmon')
            plt.title(f'{col} - Classe: {classe}')
            plt.xlabel(col)
            plt.tight_layout()
             
            plt.savefig(f"{figuras}/box_{col}_{classe}.png", bbox_inches='tight')
            plt.close()

    sumario_condicional = pd.concat(resultados)
    sumario_condicional.to_csv("resultados/tables/sumario_condicional.csv")

    #categóricas
    for col in categoricas.columns:
        tabela_freq = pd.crosstab(df[classe_col], df[col], normalize='index') * 100
        tabela_freq.plot(kind='bar', stacked=True, figsize=(6,4), colormap='viridis')
        plt.title(f'Distribuição de {col} por classe ({classe_col})')
        plt.ylabel('% dentro da classe')
        plt.xlabel(classe_col)
        plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
         
        plt.savefig(f"{figuras}/cat_{col}_por_{classe_col}.png", bbox_inches='tight')
        plt.close()

    print("Análise univariada condicional concluída!")
    return sumario_condicional