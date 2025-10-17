import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

def carregar_dados(caminho_ou_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(caminho_ou_url)
        return df
    except Exception as e:
        print(f"Ocorreu um erro ao carregar os dados: {e}")
        return None

def tratar_dados_faltantes(df: pd.DataFrame, coluna) -> pd.DataFrame:
    df_tratado = df.copy()
    if df_tratado[coluna].isnull().sum() > 0:
        moda = df_tratado[coluna].mode()[0]
        df_tratado[coluna] = df_tratado[coluna].fillna(moda)
    print(f"Valores nulos em '{coluna}' foram preenchidos com a moda: '{moda}'")
    return df_tratado

def remover_duplicatas(df: pd.DataFrame) -> pd.DataFrame:
    if df.duplicated().sum() > 0:
        df_sem_duplicatas = df.drop_duplicates(ignore_index=True)
        print(f"{duplicatas} linhas duplicadas foram removidas.")
        return df_sem_duplicatas
    else:
        print("Nenhuma linha duplicada encontrada.")
        return df

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
    print("Coluna 'performance_class' criada a partir de 'exam_score' com as novas definições de classes.")
    return df_discretizado

def obter_dicionario_de_dados() -> pd.DataFrame:
    lista_de_variaveis = [
        {"variavel": "student_id", "descricao": "Identificador único para cada estudante.", "tipo": "qualitativa", "subtipo": "nominal"},
        {"variavel": "age", "descricao": "Idade do estudante em anos.", "tipo": "quantitativa", "subtipo": "discreta"},
        {"variavel": "gender", "descricao": "Gênero do estudante.", "tipo": "qualitativa", "subtipo": "nominal"},
        {"variavel": "study_hours_per_day", "descricao": "Média de horas de estudo por dia.", "tipo": "quantitativa", "subtipo": "contínua"},
        {"variavel": "social_media_hours", "descricao": "Média de horas em redes sociais por dia.", "tipo": "quantitativa", "subtipo": "contínua"},
        {"variavel": "netflix_hours", "descricao": "Média de horas assistindo Netflix por dia.", "tipo": "quantitativa", "subtipo": "contínua"},
        {"variavel": "part_time_job", "descricao": "Indica se o estudante tem trabalho de meio período (Yes/No).", "tipo": "qualitativa", "subtipo": "nominal"},
        {"variavel": "attendance_percentage", "descricao": "Percentual de presença do estudante nas aulas.", "tipo": "quantitativa", "subtipo": "contínua"},
        {"variavel": "sleep_hours", "descricao": "Média de horas de sono por noite.", "tipo": "quantitativa", "subtipo": "contínua"},
        {"variavel": "diet_quality", "descricao": "Qualidade da dieta (Good, Fair, Poor).", "tipo": "qualitativa", "subtipo": "ordinal"},
        {"variavel": "exercise_frequency", "descricao": "Frequência de exercícios por semana.", "tipo": "quantitativa", "subtipo": "discreta"},
        {"variavel": "parental_education_level", "descricao": "Nível de escolaridade dos pais.", "tipo": "qualitativa", "subtipo": "ordinal"},
        {"variavel": "internet_quality", "descricao": "Qualidade da conexão de internet (Good, Average, Poor).", "tipo": "qualitativa", "subtipo": "ordinal"},
        {"variavel": "mental_health_rating", "descricao": "Autoavaliação da saúde mental (1 a 10).", "tipo": "quantitativa", "subtipo": "discreta"},
        {"variavel": "extracurricular_participation", "descricao": "Participa de atividades extracurriculares (Yes/No).", "tipo": "qualitativa", "subtipo": "nominal"},
        {"variavel": "exam_score", "descricao": "Pontuação no exame (variável alvo).", "tipo": "quantitativa", "subtipo": "contínua"},
        {"variavel": "performance_class", "descricao": "Classe de desempenho baseada no exam_score.", "tipo": "qualitativa", "subtipo": "ordinal"}
    ]
    df_dict = pd.DataFrame(lista_de_variaveis)
    return df_dict

def salvar_dataframe_processado(df: pd.DataFrame) -> None:
    # caminho absoluto da pasta 'dados' (a partir da raiz do projeto)
    proj_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    caminho_dados = os.path.join(proj_root, "dados", "student_habits_preprocessed.csv")

    # salva o DataFrame em CSV
    df.to_csv(caminho_dados, index=False)  # index=False para não salvar a coluna de índice

    print(f"Arquivo salvo em: {caminho_dados}")

def executar_pre_processamento_completo(caminho_ou_url: str) -> pd.DataFrame:
    df = carregar_dados(caminho_ou_url)
    if df is None:
        return None
    
    df = tratar_dados_faltantes(df)
    df = remover_duplicatas(df)
    df = discretizar_variavel_alvo(df)
    salvar_dataframe_processado()
    
    return df