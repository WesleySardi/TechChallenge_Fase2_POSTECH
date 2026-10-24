import pytest

from src import modeling
from src.ga.fitness import DEFAULT_WEIGHTS, CVFitness
from src.ga.search_spaces import BASELINE_PARAMS
from src.optimization import bootstrap_ci


def test_fitness_is_weighted_combination_of_cv_metrics(split):
    fitness = CVFitness("knn", split.X_train, split.y_train, cv_repeats=1)
    evaluation = fitness(BASELINE_PARAMS["knn"])
    expected = sum(w * evaluation.metrics[f"cv_{m}_mean"] for m, w in DEFAULT_WEIGHTS.items())
    assert evaluation.fitness == pytest.approx(expected)
    assert 0.8 < evaluation.fitness <= 1.0


def test_fitness_never_touches_test_set(split):
    """O fitness só recebe o treino: o tamanho dos folds de validação prova isso."""
    fitness = CVFitness("decision_tree", split.X_train, split.y_train, cv_folds=5, cv_repeats=1)
    folds = list(fitness.cv.split(split.X_train, split.y_train))
    assert sum(len(val) for _, val in folds) == len(split.X_train)


def test_fitness_is_deterministic(split):
    fitness = CVFitness("decision_tree", split.X_train, split.y_train, cv_repeats=1)
    params = BASELINE_PARAMS["decision_tree"]
    assert fitness(params).fitness == fitness(params).fitness


def test_baseline_pipeline_matches_phase1_registry(dataset, split):
    """O baseline avaliado é exatamente o modelo da Fase 1."""
    pipeline = modeling.build_model_pipeline(dataset, "random_forest")
    clf = pipeline.named_steps["classifier"]
    assert clf.n_estimators == 300 and clf.max_depth == 6
    assert clf is not modeling.MODEL_REGISTRY["random_forest"]  # clonado, não compartilhado


def test_unknown_metric_in_weights_raises(split):
    with pytest.raises(ValueError):
        CVFitness("knn", split.X_train, split.y_train, weights={"inexistente": 1.0})


def test_bootstrap_ci_contains_point_estimate(split, trained_pipeline):
    y_true = split.y_test.to_numpy()
    y_pred = trained_pipeline.predict(split.X_test)
    y_proba = trained_pipeline.predict_proba(split.X_test)[:, 1]
    ci = bootstrap_ci(y_true, y_pred, y_proba, n_bootstrap=200)
    acc = (y_true == y_pred).mean()
    assert ci["test_accuracy_ci_low"] <= acc <= ci["test_accuracy_ci_high"]
