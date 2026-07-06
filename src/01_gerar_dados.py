"""
Geração de dados sintéticos: clientes de uma operadora de telecom.
O objetivo é prever churn (cancelamento). Semente fixa => reprodutível.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 7000

DATA_DIR = Path(__file__).resolve().parent.parent / "dados"
DATA_DIR.mkdir(exist_ok=True)

def gerar():
    tempo_casa = RNG.integers(1, 73, N)
    idade = RNG.integers(18, 80, N)
    mensalidade = np.round(RNG.normal(70, 25, N).clip(20, 150), 2)
    tipo_contrato = RNG.choice(
        ["Mensal", "Anual", "Bianual"], N, p=[0.55, 0.30, 0.15]
    )
    suporte_tecnico = RNG.choice(["Sim", "Nao"], N, p=[0.4, 0.6])
    internet_fibra = RNG.choice(["Sim", "Nao"], N, p=[0.45, 0.55])
    reclamacoes = RNG.poisson(1.2, N)

    score = (
        -0.4
        - 0.055 * tempo_casa
        + 0.020 * (mensalidade - 70)
        + 1.30 * (tipo_contrato == "Mensal")
        - 1.10 * (tipo_contrato == "Bianual")
        - 0.90 * (suporte_tecnico == "Sim")
        + 0.45 * reclamacoes
        + RNG.normal(0, 0.35, N)
    )
    prob = 1 / (1 + np.exp(-score))
    churn = (RNG.random(N) < prob).astype(int)

    df = pd.DataFrame(
        {
            "id_cliente": np.arange(1, N + 1),
            "idade": idade,
            "tempo_casa_meses": tempo_casa,
            "tipo_contrato": tipo_contrato,
            "mensalidade": mensalidade,
            "internet_fibra": internet_fibra,
            "suporte_tecnico": suporte_tecnico,
            "reclamacoes_6m": reclamacoes,
            "churn": churn,
        }
    )
    destino = DATA_DIR / "clientes_telecom.csv"
    df.to_csv(destino, index=False)
    print(f"Dados gerados: {destino}  ({len(df)} linhas, churn={churn.mean():.1%})")
    return df

if __name__ == "__main__":
    gerar()
