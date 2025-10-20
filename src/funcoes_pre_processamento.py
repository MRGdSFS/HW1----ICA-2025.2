import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# ==========================
# CARREGAR DADOS
# ==========================
def carregar_dados(caminho_ou_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(caminho_ou_url)
        print("Dados carregados com sucesso.")
        return df
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return None

# ==========================
# TRATAR NULOS (NUMÉRICA = MÉDIA | CATEGÓRICA = MODA)
# ==========================
def tratar_dados_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    df_tratado = df.copy()

    for coluna in df_tratado.columns:
        if df_tratado[coluna].isnull().sum() > 0:
            if df_tratado[coluna].dtype in ['float64', 'int64']:
                media = df_tratado[coluna].mean()
                df_tratado[coluna] = df_tratado[coluna].fillna(media)
                print(f"Nulos em '{coluna}' preenchidos com MÉDIA: {media:.2f}")
            else:
                moda = df_tratado[coluna].mode()[0]
                df_tratado[coluna] = df_tratado[coluna].fillna(moda)
                print(f"Nulos em '{coluna}' preenchidos com MODA: '{moda}'")
    return df_tratado

# ==========================
# REMOVER DUPLICATAS
# ==========================
def remover_duplicatas(df: pd.DataFrame) -> pd.DataFrame:
    duplicatas = df.duplicated().sum()
    if duplicatas > 0:
        df_sem_duplicatas = df.drop_duplicates(ignore_index=True)
        print(f"{duplicatas} linhas duplicadas foram removidas.")
        return df_sem_duplicatas
    else:
        print("Nenhuma linha duplicada encontrada.")
        return df

# ==========================
# REMOVER OUTLIERS
# ==========================
def remover_outliers(df: pd.DataFrame, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    df_sem_outliers = df[(df[coluna] >= lim_inf) & (df[coluna] <= lim_sup)]
    return df_sem_outliers

# ==========================
# DISCRETIZAR VARIÁVEL ALVO
# ==========================
def discretizar_variavel_alvo(df: pd.DataFrame) -> pd.DataFrame:
    df_discretizado = df.copy()
    bins = [0, 40, 70, 90, 101]
    labels = ['Reprovado', 'Recuperação', 'Bom', 'Excelente']
    df_discretizado['performance_class'] = pd.cut(
        df_discretizado['exam_score'], 
        bins=bins, 
        labels=labels, 
        right=False
    )
    print("Coluna 'performance_class' criada a partir de 'exam_score'.")
    return df_discretizado

# ==========================
# DICIONÁRIO DE DADOS
# ==========================
def obter_dicionario_de_dados() -> pd.DataFrame:
    lista_de_variaveis = [
        {"variavel": "student_id", "descricao": "Identificador único para cada estudante.", "tipo": "qualitativa", "subtipo": "nominal"},
        {"variavel": "age", "descricao": "Idade do estudante.", "tipo": "quantitativa", "subtipo": "discreta"},
        {"variavel": "gender", "descricao": "Gênero do estudante.", "tipo": "qualitativa", "subtipo": "nominal"},
        {"variavel": "study_hours_per_day", "descricao": "Horas de estudo por dia.", "tipo": "quantitativa", "subtipo": "contínua"},
        {"variavel": "social_media_hours", "descricao": "Horas em redes sociais.", "tipo": "quantitativa", "subtipo": "contínua"},
        {"variavel": "netflix_hours", "descricao": "Horas assistindo Netflix.", "tipo": "quantitativa", "subtipo": "contínua"},
        {"variavel": "part_time_job", "descricao": "Trabalho de meio período (Yes/No).", "tipo": "qualitativa", "subtipo": "nominal"},
        {"variavel": "attendance_percentage", "descricao": "Presença nas aulas.", "tipo": "quantitativa", "subtipo": "contínua"},
        {"variavel": "sleep_hours", "descricao": "Horas de sono.", "tipo": "quantitativa", "subtipo": "contínua"},
        {"variavel": "diet_quality", "descricao": "Qualidade da dieta.", "tipo": "qualitativa", "subtipo": "ordinal"},
        {"variavel": "exercise_frequency", "descricao": "Frequência de exercícios.", "tipo": "quantitativa", "subtipo": "discreta"},
        {"variavel": "parental_education_level", "descricao": "Escolaridade dos pais.", "tipo": "qualitativa", "subtipo": "ordinal"},
        {"variavel": "internet_quality", "descricao": "Qualidade da internet.", "tipo": "qualitativa", "subtipo": "ordinal"},
        {"variavel": "mental_health_rating", "descricao": "Saúde mental (1-10).", "tipo": "quantitativa", "subtipo": "discreta"},
        {"variavel": "extracurricular_participation", "descricao": "Participa de extracurriculares.", "tipo": "qualitativa", "subtipo": "nominal"},
        {"variavel": "exam_score", "descricao": "Pontuação no exame (alvo).", "tipo": "quantitativa", "subtipo": "contínua"},
        {"variavel": "performance_class", "descricao": "Classe de desempenho.", "tipo": "qualitativa", "subtipo": "ordinal"}
    ]
    return pd.DataFrame(lista_de_variaveis)

# ==========================
# SALVAR ARQUIVO PROCESSADO
# ==========================
def salvar_dataframe_processado(df: pd.DataFrame) -> None:
    proj_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    caminho_dados = os.path.join(proj_root, "dados", "student_habits_preprocessed.csv")
    df.to_csv(caminho_dados, index=False)
    print(f"Arquivo salvo em: {caminho_dados}")

# ==========================
# PIPELINE COMPLETO
# ==========================
def executar_pre_processamento_completo(caminho_ou_url: str) -> pd.DataFrame:
    df = carregar_dados(caminho_ou_url)
    if df is None:
        return None
    
    df = tratar_dados_faltantes(df)
    df = remover_duplicatas(df)
    df = discretizar_variavel_alvo(df)
    salvar_dataframe_processado(df)
    
    return df
