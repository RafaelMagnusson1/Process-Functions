import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go

def time_convert(df):

    df['Time'] = pd.to_datetime(df['Time']).dt.floor('S')
    df.set_index('Time', inplace=True)
    return df

def vazoes_append(df,rota,vazao):

    vazoes = []
    for i in range(80,840,10):
        df_filtrado = df.loc[df[rota] == i]
        media_vazao_cip = df_filtrado[vazao].mean()
        vazoes.append(media_vazao_cip)
    
    return vazoes

def vazoes_std(df,rota,vazao):

    vazões_desvpad = []
    for i in range(80,840,10):
        df_filtrado = df.loc[df[rota] == i]
        media_vazao_cip = df_filtrado[vazao].std()
        vazões_desvpad.append(media_vazao_cip)
    
    return vazões_desvpad


def define_df(vazoes):

    passos = []
    for i in range(80,840,10):
        passos.append(i)

    df__cip = {"Passos":passos,"Vazões médias":vazoes}
    df_passos = pd.DataFrame(df__cip)
    return df_passos

def plot_vazoes(df_passos, mf):

    df_SP=pd.read_csv(r'C:\Users\Rafael Magnusson\Desktop\VSCode\Sterility_Hold_L3\SP_passos.csv',sep=",",encoding="latin1")
    df_merged_mf32 = pd.merge(df_passos,df_SP, left_index=True, right_index=True, how='inner')

    import matplotlib.patches as mpatches
    df_merged_mf32['Vazões médias'].fillna(0, inplace=True)

    plt.figure(figsize=(50, 10))  # Aumentar o tamanho da figura
    bar_plot = plt.bar(df_merged_mf32["Passos"], df_merged_mf32["Vazões médias"], color="purple", width=5.0)

    line_plot, = plt.plot(df_merged_mf32["Passos"], df_merged_mf32["SP"], color="red", linestyle="--", linewidth=2.5)

    plt.xticks(range(0, df_merged_mf32["Passos"].max() + 1, 10), fontsize=16)
    plt.tick_params(axis='x', length=10, width=1, direction='inout', bottom=True)

    for x, y in zip(df_merged_mf32["Passos"], df_merged_mf32["Vazões médias"]):
        plt.text(x, y, str(int(y)), fontsize=20, ha='center', va='bottom')

    plt.xlabel("Passos da rota de CIP", fontsize=25)
    plt.ylabel("Vazão (m3/h)", fontsize=25)
    plt.title(f"Vazão média CIP {mf} por passo da rota", fontsize=20)

    bar_patch = mpatches.Patch(color="purple", label="Vazão média")
    line_patch = mpatches.Patch(color="red", linestyle="--", linewidth=2.5, label="SP de projeto")
    plt.legend(handles=[bar_patch, line_patch])

    plt.show()

def plot_nvl(df,rota,nivel):

    ax = plt.figure(figsize = (20,14))
    plt.scatter(df[rota],df[nivel])
    plt.xlim(80,840)
    plt.xticks(range(80, 841, 10), rotation = 90)
    plt.title("Nível do fermentador ao longo do CIP")
    plt.xlabel("Passo da rota de CIP")
    plt.ylabel("Nível (%)")
    plt.show()


#Funções referentes à análise de TAT do CIP:

def generate_cip_series(df, rota):

    df_cip_sem_hold = df[(df[rota] > 20) & (df[rota] < 850)]
    df_cip_sem_hold_passos = df_cip_sem_hold[rota]
    series_tat = df_cip_sem_hold_passos.drop_duplicates(keep='first')

    return series_tat

def calculate_time_diff(series_tat):

    diferencas_de_tempo = series_tat.index.to_series().diff().dropna()
    diferencas_de_tempo_em_minutos = diferencas_de_tempo.dt.total_seconds() / 60
    lista_diferencas_de_tempo = diferencas_de_tempo_em_minutos.tolist()

    return lista_diferencas_de_tempo

def generate_passos_series(series_tat):

    lista_tat = list(series_tat)
    lista_tat.pop(-1)

    return lista_tat


def plot_step_time(lista_tat, lista_diferencas_de_tempo, MF):
    
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=lista_tat,
        y=lista_diferencas_de_tempo,
        marker_color='purple',
        text=lista_diferencas_de_tempo,
        textposition='outside',  
        texttemplate='%{y:.1f}',  
    ))


    fig.update_layout(
        title= f"Tempo de cada passo CIP - MF {MF}",
        xaxis=dict(
            title="Passos da rota de CIP",
            tickmode='array',
            tickvals=lista_tat,
            ticktext=[str(x) for x in lista_tat],  
            tickangle=90, 
        ),
        yaxis=dict(
            title="Tempo (min)",
        ),
        width=1000,  
        height=600, 
    )

    fig.show()

