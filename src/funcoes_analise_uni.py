# Neste arquivo temos funções reutilizáveis para análise univariada incondicional e condicionada por classe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import math
from IPython.display import display

import funcoes_pre_processamento as pp

# Esta função irá separar nossas variáveis entre quantitativas e qualitativas,
# ela tem como argumento o dataframe que contém o dicionário de dados e retorna uma tupla (nesse caso, duas listas).
def separar_variaveis(df_dict: pd.DataFrame) -> tuple:
    
    # Usamos o dataframe do dicionário (df_dict) feito no pré processamento para criar
    # uma máscara booleana onde iremos filtrar as linhas onde o 'tipo' seja 'quantitativo' e depois pegue
    # só a coluna 'variavel' dessas linhas filtradas e converta elas para uma lista python com a função 'tolist()'
    # jogando tudo isso numa variável 'quantitativa', fazemos o mesmo processo para as variáveis que têm tipo.
    quantitativas = df_dict[df_dict['tipo'] == 'quantitativa']['variavel'].tolist()
    qualitativas = df_dict[df_dict['tipo'] == 'qualitativa']['variavel'].tolist()


    # Aqui, tiramos as variáveis que não serão usadas na análise univariada, ou seja, a que não contribui como preditor (student_id), 
    # a nossa variável alvo (exam_score) e a que será usada na análise condicional(perfomance_class). 
    if 'student_id' in qualitativas:
        qualitativas.remove('student_id')

    if 'exam_score' in quantitativas:
        quantitativas.remove('exam_score')
     
    if 'performance_class' in qualitativas:
        qualitativas.remove('performance_class')
        
    # Por fim, retornamos a tupla contendo duas listas de variáveis, a quantitativa e a qualitativa.   
    return quantitativas, qualitativas

# Esta função irá fazer alguns cálculos utlizando as funções da biblioteca do pandas nas colunas do dataframe que indicarmos como parâmetro.
def estatisticas_descritivas_quantitativas(df: pd.DataFrame, colunas: list) -> pd.DataFrame:
     
     # Vamos usar a função agg da pandas para realizar 1-contagem dos valores, 2-encontrar o mínimo, 3-encontrar o máximo,4-calcular a média, 5-calcular a mediana,
     # 6-calcular o desvio padrão, 7-calcular a skewness (assimetria). Depois, usa o .T pra transpostar a coluna pra uma linha para deixar mais agradável aos olhos.
     estatisticas = df[colunas].agg(['count', 'min', 'max', 'mean', 'median', 'std', 'skew']).T

    # Mudando os nomes para português:
     estatisticas.columns = ['Contagem', 'Mínimo', 'Máximo', 'Média', 'Mediana', 'Desvio Padrão', 'Assimetria']

    #Por fim, retornamos essas estatísticas:
     return estatisticas 

# Esta função irá fazer algo parecido com a da anterior só que dessa vez será com as variáveis qualitativas.
# Usaremos como parâmetros a tabela completa (df) e uma lista com os nomes das colunas qualitativas que queremos ver(colunas), retornando um dicionário(dict) jpa que cada coluna tem categorias diferentes, então não retornaremos um dataframe como na função anterior.
def estatisticas_descritivas_qualitativas(df: pd.DataFrame, colunas: list) -> dict:
     
     resultados = {} # Dicionário vazio
     
     # col vai passar pelas colunas do dataframe e a função value_conts vai contar cada valor único.
     # o normalize true vai nos dar frações entre 0 e 1 e depois passar pra porcentagem com o *100 pra termos nossas medidas diferentes padronizadas e normalizadas em forma de fração. É uma forma de discretizar as nossas variáveis categóricas (qualitativas.
     for col in colunas:
        contagem = df[col].value_counts()
        proporcao = df[col].value_counts(normalize=True) * 100
        df_temp = pd.DataFrame({
            'Contagem': contagem,
            'Proporção (%)': proporcao
        })
        resultados[col] = df_temp
     
     #Ao final teremos um dicionário de dataframes onde cada dataframe tem uma contagem e uma proporção, e cada dataframe está associado a uma coluna que veio do parâmetro da função.
     return resultados

#  Aqui vamos calcular estatísticas descritivas (média, desvio padrão e assimetria) para cada variável quantitativa para cada classe de desempenho.
def estatisticas_condicionais_por_classe_especifica(df: pd.DataFrame, colunas_quantitativas: list, classes: list, coluna_classe: str = 'performance_class') -> pd.DataFrame:
    """
    Calcula estatísticas descritivas para variáveis quantitativas APENAS para as classes especificadas.
    
    Parâmetros:
    df: DataFrame com os dados
    colunas_quantitativas: Lista de variáveis quantitativas
    classes: Lista das classes específicas que você quer analisar (ex: ['Reprovado', 'Recuperação'])
    coluna_classe: Nome da coluna que define as classes (padrão: 'performance_class')
    
    Retorna:
    DataFrame com estatísticas apenas para as classes solicitadas
    """
    resultados = []
    
    # Para cada variável quantitativa
    for var in colunas_quantitativas:
        # Para cada classe ESPECÍFICA fornecida no parâmetro
        for classe in classes:
            # Filtra os dados para a classe específica
            dados_classe = df[df[coluna_classe] == classe][var].dropna()
            
            # Calcula as estatísticas
            n_obs = len(dados_classe)
            media = dados_classe.mean()
            desvio_padrao = dados_classe.std()
            assimetria = dados_classe.skew()
            
            # Adiciona ao resultado
            resultados.append({
                'Variável': var,
                'Classe': classe,
                'N_observações': n_obs,
                'Média': media,
                'Desvio_Padrão': desvio_padrao,
                'Assimetria': assimetria
            })
    
    return pd.DataFrame(resultados)

#Esta função vai plotar vários gráficos para vermos de uma vez
def plot_matriz_univariada_quantitativa(df: pd.DataFrame, colunas: list) -> None:
    
    n_vars = len(colunas) 
    n_plots = n_vars * 2 # Cada variável quantitativa tem 1 histograma e 1 boxplot, logo o número de plots é vezes 2.

    n_cols = 4 # Vamos limitar o número de colunas em 4 para questões de legibilidade de cada subplot.
    n_rows = math.ceil(n_plots / n_cols) # Essa função ceil arredonda o número de linhas que a gente vai ter para cima. 

    # Aqui vamos ter a criação da figura e dos subplots que ela tem com um tamanho 4 por 3.
    fig, subs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))      

    # Só que se tiver uma matriz o trabalho pra fazer os boxplots e histogramas pode ser muito complicado,
    # para isso vamos usar a função flatten() nos subplots e facilitar nossa vida.
    subs = subs.flatten() # Matriz vira lista.  

    # Vamos usar as posições ímpares dos subplots (subs) para formar os boxplots e as pares os histogramas.
    # Já que agora temos uma lista dos subplots, podemos utilizar as funções da biblioteca seaborn histplot e boxplot para mostrar nossos gráficos queridos.
    for i, coluna in enumerate(colunas): # Faz o histograma e boxplot pra cada coluna que veio do parâmetro.
        # Histograma (distribuição).
        sns.histplot(df[coluna], kde=True, ax=subs[i*2]) # kde=true desenha a linha em cima das barras. kde é o nosso kernel density estimate que vimos em sala de aula que mostra a distribuição como um formato geral e suave.
        subs[i*2].set_title(f'Hist: {coluna}', fontsize=10)
        subs[i*2].set_xlabel(coluna, fontsize=8)
        
        # Boxplot (outliers e medidas de posição).
        sns.boxplot(x=df[coluna], ax=subs[i*2 + 1]) # Esse último parâmetro é só pra mostrar onde ele vai desenhar.
        subs[i*2 + 1].set_title(f'Box: {coluna}', fontsize=10)
        subs[i*2 + 1].set_xlabel(coluna, fontsize=8)

    # Não devemos esquecer de quando costruímos nossa figura com mais lugares para gráficos do que deveríamos, por isso precisamos removê-los
    for j in range (n_plots, n_rows*n_cols): # Esse range vai pegar o excesso de lugares que não possuem subplots e com a função do matplotlib vai remover esses lugares e nossa figura ficará mais bonita.
        fig.delaxes(subs[j])

    # Por fim, plotando:
    plt.suptitle ('Análise Univariada Incondicional - Variáveis Quantitativas (Histograma e Boxplot)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
            
# Agora vamos fazer a função de plot de matriz das qualitativas:
def plot_matriz_univariada_qualitativa(df: pd.DataFrame, colunas: list) -> None:
    
    # A lógica aqui vai ser parecida com a da função anterior, só que gerando gráficos de barras para as variáveis qualitativas.

    n_vars = len(colunas)
    n_plots = n_vars
    
    # Mudamos um pouco do layout, vamos deixar 3 colunas aqui.
    n_cols = 3
    n_rows = math.ceil(n_plots / n_cols)
    
    fig, subs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    subs = subs.flatten()

    for i, coluna in enumerate(colunas):
        
        # Vamos usar a função novamente do value_count já que o barplot mostra a frequência de cada categoria.
        contagens = df[coluna].value_counts()
        sns.barplot(x=contagens.index, y=contagens.values, ax=subs[i]) # .index = nome das barras, .values = altura delas.
        subs[i].set_title(f'Distribuição de {coluna}', fontsize=10)
        subs[i].set_xlabel(coluna, fontsize=8)
        subs[i].set_ylabel('Contagem', fontsize=8)
        subs[i].tick_params(axis='x', rotation=45) # Pra deixar organizado e não tudo um atrás do outro e caber no gráfico.

    # Removendo os não utilizados:
    for j in range(n_plots, n_rows * n_cols):
        fig.delaxes(subs[j])

    #Por fim plotando:    
    plt.suptitle('Análise Univariada Incondicional - Variáveis Qualitativas (Gráfico de Barras)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

# Agora vamos implementar as funções da análise univariada condicional (às categorias reprovado, recuperação, bom e excelente)  
def plot_matriz_condicional_quantitativa(df: pd.DataFrame, colunas_quant: list, coluna_alvo: str) -> None: # Aqui passamos como parâmetro o nome da coluna alvo como a perfomance_class.

    #Vamos mudar um pouco da lógica feita nas incondicionais devido a quantidade de gráficos que irão aparecer aqui, se usassemos o flatten() ficaria muito confuso,
    # já que a gente perderia a noção visual por linha e coluna então é melhor uma outra abordagem:

    # Retorna valores únicos da coluna alvo garantindo ordem consistente das classes.
    classes = list(df[coluna_alvo].dropna().unique())

    for coluna in colunas_quant:
        # Cria a figura de 1 linha para boxplot e 1 linha pra histogramas.
        fig = plt.figure(figsize=(16, 8))
            
        # Boxplots.
        subs1 = plt.subplot2grid((2, len(classes)), (0, 0), colspan=len(classes))
        sns.boxplot(x=coluna_alvo, y=coluna, data=df, ax=subs1, order=classes)
        subs1.set_title(f'{coluna} por {coluna_alvo}')
        subs1.set_xlabel(coluna_alvo)
        subs1.set_ylabel(coluna)

        # Histogramas.
        for i, classe in enumerate(classes):
            subs = plt.subplot2grid((2, len(classes)), (1, i))  # Função que permite o posicionamento preciso.
            subset = df[df[coluna_alvo] == classe][coluna].dropna()
            if subset.empty:
                # evita erro se não houver dados para uma classe
                subs.set_visible(False)
                continue
            subs.hist(subset, bins=10, density=True)
            sns.kdeplot(subset, ax=subs)
            subs.set_title(f'{coluna} - {classe}')
            subs.set_xlabel(coluna)
            subs.set_ylabel('Densidade')

        plt.suptitle(f'Análise Condicional - {coluna}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

# Agora a qualitativa condicional:
def plot_matriz_condicional_qualitativa(df: pd.DataFrame, colunas_qual: list, coluna_alvo: str) -> None:
    
    n_vars = len(colunas_qual)
    n_plots = n_vars
    
    n_cols = 3
    n_rows = math.ceil(n_plots / n_cols)
    
    fig,subs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    subs =subs.flatten()
    
    for i, coluna in enumerate(colunas_qual):
        
        # Para a qualitativa, vamos usar a função countplot, que junto com o hue (separação em cores) vai separar a coluna alvo em cores, 
        # o countplot conta quantos cada tipo de desempenho existe dentro daquela categoria.
        sns.countplot(x=coluna, hue=coluna_alvo, data=df, ax=subs[i])
        subs[i].set_title(f'{coluna} por {coluna_alvo}', fontsize=10)
        subs[i].set_xlabel(coluna, fontsize=8)
        subs[i].set_ylabel('Contagem', fontsize=8)
        subs[i].tick_params(axis='x', rotation=45)
        subs[i].legend(title=coluna_alvo, fontsize=8)

    # Removendo eixos. 
    for j in range(n_plots, n_rows * n_cols):
        fig.delaxes(subs[j])
        
    plt.suptitle(f'Análise Univariada Condicional - Qualitativas vs {coluna_alvo}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

# Esta função vai realizar a análise univariada incondicional e condicional, separando as variáveis, calculando as estatísticas e plotando as matrizes de gráficos desta análise.
# Recebe como parâmetros o dataframe processados, o dicionário de dados e o nome da variável alvo pra análise condicional caso você queira ver tudo de uma vez.
def executar_analise_univariada_completa(df: pd.DataFrame, df_dict: pd.DataFrame, coluna_alvo: str = 'performance_class') -> None:

    quantitativas, qualitativas = separar_variaveis(df_dict)
    
    print("--- 1. Análise Univariada Incondicional (Distribuição Pura) ---")
    
    print("\n[TABELA] Estatísticas Descritivas para Variáveis Quantitativas:")

    display(estatisticas_descritivas_quantitativas(df, quantitativas))
    
    print("\n[TABELAS] Estatísticas Descritivas para Variáveis Qualitativas (Contagem e Proporção):")
    
    desc_qual = estatisticas_descritivas_qualitativas(df, qualitativas)
    
    for col, tabela in desc_qual.items():
        print(f"\nVariável: {col}")
        display(tabela)
        
    
    print("\n[GRÁFICOS] Matriz de Gráficos Univariados Quantitativos (Histograma e Boxplot):")
    # Aqui, teremos uma matriz com gráficos.
    plot_matriz_univariada_quantitativa(df, quantitativas)
    
    
    print("\n[GRÁFICOS] Matriz de Gráficos Univariados Qualitativos (Gráfico de Barras):")
    # Aqui, teremos gráficos.
    plot_matriz_univariada_qualitativa(df, qualitativas)
    
    print("\n--- 2. Análise Univariada Condicional (Relação com a Variável Alvo) ---")
    
    print(f"\n[GRÁFICOS] Matriz de Boxplots Condicionais (Variáveis Quantitativas vs {coluna_alvo}):")
    # Gera a matriz de boxplots.
    plot_matriz_condicional_quantitativa(df, quantitativas, coluna_alvo)
    
    print(f"\n[GRÁFICOS] Matriz de Gráficos de Contagem Condicionais (Variáveis Qualitativas vs {coluna_alvo}):")
    # Gera a matriz de countplots.
    plot_matriz_condicional_qualitativa(df, qualitativas, coluna_alvo)
