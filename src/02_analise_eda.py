"""Análise exploratória (EDA) do dataset de churn. Salva gráficos em /imagens."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

sns.set_theme(style="whitegrid")
BASE = Path(__file__).resolve().parent.parent
IMG = BASE / "imagens"
IMG.mkdir(exist_ok=True)

def carregar():
    return pd.read_csv(BASE / "dados" / "clientes_telecom.csv")

def main():
    df = carregar()
    print("Resumo estatístico:\n", df.describe().round(2), "\n")
    print("Taxa de churn:", f"{df['churn'].mean():.1%}")

    fig, ax = plt.subplots(figsize=(7, 4))
    taxa = df.groupby("tipo_contrato")["churn"].mean().sort_values()
    sns.barplot(x=taxa.index, y=taxa.values, ax=ax, palette="rocket")
    ax.set(title="Taxa de churn por tipo de contrato", ylabel="Taxa de churn")
    for i, v in enumerate(taxa.values):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center")
    fig.tight_layout(); fig.savefig(IMG / "churn_por_contrato.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="churn", y="tempo_casa_meses", ax=ax, palette="crest")
    ax.set(title="Tempo de casa x Churn", xlabel="Churn (0=ficou, 1=saiu)")
    fig.tight_layout(); fig.savefig(IMG / "tempo_casa_churn.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    num = df.drop(columns=["id_cliente"]).select_dtypes("number")
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set(title="Matriz de correlação")
    fig.tight_layout(); fig.savefig(IMG / "correlacao.png", dpi=110); plt.close(fig)

    print(f"Gráficos salvos em {IMG}")

if __name__ == "__main__":
    main()
