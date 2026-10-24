"""Limpeza de dados e pipeline de pré-processamento em scikit-learn."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NON_FEATURE_COLS = {"diagnosis", "diagnosis_label"}


@dataclass
class CleaningReport:
    n_rows_before: int
    n_rows_after: int
    n_duplicates_removed: int
    missing_values_by_column: pd.Series


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Remove duplicatas e reporta valores ausentes (sem descartar linhas por NaN,
    pois o pipeline de pré-processamento trata isso via imputação)."""
    n_before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    missing = df_clean.isna().sum()

    report = CleaningReport(
        n_rows_before=n_before,
        n_rows_after=len(df_clean),
        n_duplicates_removed=n_before - len(df_clean),
        missing_values_by_column=missing[missing > 0],
    )
    return df_clean, report


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def build_preprocessing_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """Monta um pipeline genérico: imputação + escala para numéricas,
    imputação + one-hot para categóricas. Neste dataset todas as features
    são numéricas, mas o pipeline permanece genérico caso variáveis
    categóricas sejam adicionadas no futuro (ex.: outro dataset de saúde)."""
    feature_cols = get_feature_columns(df)
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = get_feature_columns(df)
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    return df[numeric_cols].corr()


def top_correlated_pairs(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Lista pares de features (sem repetição) com correlação absoluta >= threshold."""
    corr = correlation_matrix(df)
    upper_triangle_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(upper_triangle_mask).stack()
    pairs = pairs[pairs.abs() >= threshold]
    result = (
        pairs.reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "correlation"})
        .sort_values("correlation", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )
    return result
