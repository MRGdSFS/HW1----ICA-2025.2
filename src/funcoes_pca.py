import pandas as pd
import matplotlib.pyplot as plt
import mpl_axes_aligner


def biplot(df_pca: pd.DataFrame, df_loads: pd.DataFrame) -> None:

    fig,ax = plt.subplots(figsize=(15,8))

    ax.scatter(df_pca.PC1.values,df_pca.PC2.values, color='b')
    ax.set_xlabel("PC1", fontsize = 10)
    ax.set_ylabel("PC2", fontsize = 10)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.axvline(0, color='gray', linewidth=0.8)


    ax2 = ax.twinx().twiny()

    font = {'color': 'g',
            'weight': 'bold',
            'size': 5,
            }
    
    colunas = {
    "study_hours_per_day": "Horas de Estudo", "social_media_hours": "Redes Sociais", "netflix_hours": "Netflix",
    "attendance_percentage": "Presença(%)", "sleep_hours": "Horas de Sono", "exercise_frequency": "Frequência de Exerc.",
    "mental_health_rating": "Saúde Mental", "gender_Male": "Gênero (Masc)", "gender_Other": "Gênero (Outro)",
    "part_time_job_Yes": "Trabalho meio período", "diet_quality_Good": "Dieta (Boa)", "diet_quality_Poor": "Dieta (Ruim)",
    "parental_education_level_High School": "Pais (Ens. Médio)", "parental_education_level_Master": "Pais (Mestrado)",
    "internet_quality_Good": "Internet (Boa)", "internet_quality_Poor": "Internet (Ruim)",
    "extracurricular_participation_Yes": "Extracurricular", "age": "Idade"
    
    }
    for col in df_loads.columns.values:
        tipx = df_loads.loc['PC1',col]
        tipy = df_loads.loc['PC2',col]

        ax2.arrow(0, 0, tipx, tipy, color = 'r', alpha = 0.5)
        ax2.text(tipx*1.05, tipy*1.05, colunas[col], fontdict = font, ha = 'center', va = 'center')
    
    mpl_axes_aligner.align.xaxes(ax, 0, ax2, 0, 0.5)
    mpl_axes_aligner.align.yaxes(ax, 0, ax2, 0, 0.5)

    plt.title("Biplot")
    plt.savefig("resultados/figuras_pca/biplot.png")