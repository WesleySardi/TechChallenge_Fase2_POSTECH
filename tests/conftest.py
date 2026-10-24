import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import data_loader, modeling, preprocessing


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    df, _ = preprocessing.clean_dataset(data_loader.load_dataset())
    return df


@pytest.fixture(scope="session")
def split(dataset) -> modeling.SplitData:
    return modeling.train_test_split_dataset(dataset)


@pytest.fixture(scope="session")
def trained_pipeline(dataset, split) -> Pipeline:
    pipeline = Pipeline([
        ("preprocessor", preprocessing.build_preprocessing_pipeline(dataset)),
        ("classifier", LogisticRegression(max_iter=5000)),
    ])
    return pipeline.fit(split.X_train, split.y_train)
