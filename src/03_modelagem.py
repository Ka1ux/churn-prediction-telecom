"""
Modelagem de churn: compara Regressão Logística e Random Forest.
Gera métricas, matriz de confusão e importância das variáveis.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay,
)

sns.set_theme(style="whitegrid")
BASE = Path(__file__).resolve().parent.parent
IMG = BASE / "imagens"; IMG.mkdir(exist_ok=True)

NUM = ["idade", "tempo_casa_meses", "mensalidade", "reclamacoes_6m"]
CAT = ["tipo_contrato", "internet_fibra", "suporte_tecnico"]

def preparar():
    df = pd.read_csv(BASE / "dados" / "clientes_telecom.csv")
    X, y = df[NUM + CAT], df["churn"]
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

def montar_pipeline(modelo):
    pre = ColumnTransformer(
        [("num", StandardScaler(), NUM),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)]
    )
    return Pipeline([("pre", pre), ("modelo", modelo)])

def main():
    Xtr, Xte, ytr, yte = preparar()
    modelos = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    melhor, melhor_auc, melhor_nome = None, 0, ""
    for nome, m in modelos.items():
        pipe = montar_pipeline(m)
        pipe.fit(Xtr, ytr)
        prob = pipe.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, prob)
        print(f"\n=== {nome} (AUC={auc:.3f}) ===")
        print(classification_report(yte, pipe.predict(Xte), digits=3))
        if auc > melhor_auc:
            melhor, melhor_auc, melhor_nome = pipe, auc, nome

    print(f"\n>>> Melhor modelo: {melhor_nome} (AUC={melhor_auc:.3f})")

    cm = confusion_matrix(yte, melhor.predict(Xte))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Ficou", "Saiu"], yticklabels=["Ficou", "Saiu"])
    ax.set(title=f"Matriz de confusão — {melhor_nome}", xlabel="Previsto", ylabel="Real")
    fig.tight_layout(); fig.savefig(IMG / "matriz_confusao.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(melhor, Xte, yte, ax=ax)
    ax.set(title=f"Curva ROC — {melhor_nome}")
    fig.tight_layout(); fig.savefig(IMG / "curva_roc.png", dpi=110); plt.close(fig)

    print(f"Gráficos de avaliação salvos em {IMG}")

if __name__ == "__main__":
    main()
