"""Carregamento do dataset Breast Cancer Wisconsin (Diagnostic)."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "breast_cancer_wisconsin.csv"

TARGET_COL = "diagnosis"
TARGET_NAMES = {0: "malignant", 1: "benign"}


def load_raw_dataframe() -> pd.DataFrame:
    """Carrega o dataset a partir do scikit-learn e retorna um DataFrame único.

    O target original do sklearn usa 0 = malignant, 1 = benign. Mantenho essa
    codificação e exponho também a versão textual em `diagnosis_label`.
    """
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.rename(columns={"target": TARGET_COL})
    df["diagnosis_label"] = df[TARGET_COL].map(TARGET_NAMES)
    return df


def save_dataset_csv(df: pd.DataFrame | None = None) -> Path:
    """Persiste o dataset em CSV dentro de data/, satisfazendo o requisito de entrega."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = load_raw_dataframe()
    df.to_csv(CSV_PATH, index=False)
    return CSV_PATH


def load_dataset(from_csv: bool = True) -> pd.DataFrame:
    """Ponto único de acesso ao dataset: usa o CSV local se existir, senão gera."""
    if from_csv and CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    df = load_raw_dataframe()
    save_dataset_csv(df)
    return df


if __name__ == "__main__":
    path = save_dataset_csv()
    print(f"Dataset salvo em: {path}")
