import numpy as np
import pytest

from src.ga.engine import Evaluation, GAConfig, GeneticAlgorithm, random_search
from src.ga.genes import CategoricalGene, FloatGene, IntGene, SearchSpace, params_key
from src.ga.operators import (
    blend_crossover,
    gaussian_mutation,
    random_reset_mutation,
    roulette_selection,
    single_point_crossover,
    tournament_selection,
    uniform_crossover,
)
from src.ga.search_spaces import BASELINE_PARAMS, SEARCH_SPACES, build_estimator


# ------------------------------------------------------------------ genes

@pytest.mark.parametrize("u", [0.0, 0.3, 0.5, 0.9999999])
def test_int_gene_decodes_within_bounds(u):
    gene = IntGene("x", 2, 20)
    assert 2 <= gene.decode(u) <= 20


def test_int_gene_covers_all_values_uniformly():
    gene = IntGene("x", 1, 4)
    values = [gene.decode(u) for u in np.linspace(0, 0.999999, 1000)]
    assert set(values) == {1, 2, 3, 4}


def test_float_gene_log_scale():
    gene = FloatGene("C", 1e-3, 1e3, log=True)
    assert gene.decode(0.0) == pytest.approx(1e-3)
    assert gene.decode(0.5) == pytest.approx(1.0)
    assert gene.decode(gene.encode(10.0)) == pytest.approx(10.0)


def test_categorical_gene_roundtrip():
    gene = CategoricalGene("cw", (None, "balanced"))
    for value in gene.choices:
        assert gene.decode(gene.encode(value)) == value


@pytest.mark.parametrize("model_name", list(SEARCH_SPACES))
def test_baseline_params_roundtrip(model_name):
    """Os hiperparâmetros da Fase 1 precisam ser representáveis (semente da população)."""
    space = SEARCH_SPACES[model_name]
    decoded = space.decode(space.encode(BASELINE_PARAMS[model_name]))
    for key, value in BASELINE_PARAMS[model_name].items():
        assert decoded[key] == pytest.approx(value) if isinstance(value, float) else decoded[key] == value


@pytest.mark.parametrize("model_name", list(SEARCH_SPACES))
def test_random_genomes_build_valid_estimators(model_name):
    space = SEARCH_SPACES[model_name]
    rng = np.random.default_rng(0)
    for _ in range(10):
        estimator = build_estimator(model_name, space.decode(space.random_genome(rng)))
        assert hasattr(estimator, "fit")


def test_params_key_is_order_independent():
    assert params_key({"a": 1, "b": 0.5}) == params_key({"b": 0.5, "a": 1})


# ------------------------------------------------------------------ operadores

def test_tournament_selection_with_full_tournament_picks_best():
    fitnesses = np.array([0.1, 0.9, 0.5])
    rng = np.random.default_rng(0)
    assert tournament_selection(fitnesses, rng, k=3) == 1


def test_roulette_prefers_fitter_individuals():
    fitnesses = np.array([0.90, 0.91, 0.99])
    rng = np.random.default_rng(0)
    picks = [roulette_selection(fitnesses, rng) for _ in range(2000)]
    assert np.bincount(picks, minlength=3).argmax() == 2


@pytest.mark.parametrize("crossover", [uniform_crossover, blend_crossover, single_point_crossover])
def test_crossover_keeps_shape_and_bounds(crossover):
    rng = np.random.default_rng(0)
    p1, p2 = rng.random(6), rng.random(6)
    c1, c2 = crossover(p1, p2, rng)
    for child in (c1, c2):
        assert child.shape == p1.shape
        assert np.all((child >= 0) & (child < 1))


def test_uniform_crossover_genes_come_from_parents():
    rng = np.random.default_rng(0)
    p1, p2 = np.zeros(10), np.full(10, 0.5)
    c1, c2 = uniform_crossover(p1, p2, rng)
    assert set(np.unique(np.concatenate([c1, c2]))) <= {0.0, 0.5}
    np.testing.assert_allclose(c1 + c2, p1 + p2)


def test_mutation_rate_zero_is_identity_and_one_changes_all():
    rng = np.random.default_rng(0)
    g = rng.random(8)
    np.testing.assert_array_equal(gaussian_mutation(g, rng, rate=0.0, sigma=0.2), g)
    assert np.all(random_reset_mutation(g, rng, rate=1.0) != g)


# ------------------------------------------------------------------ engine

TOY_SPACE = SearchSpace([IntGene("a", 0, 50), FloatGene("b", 0.0, 1.0), CategoricalGene("c", ("x", "y", "z"))])


def toy_fitness(params):
    """Ótimo conhecido: a=37, b=0.8, c='y'."""
    score = -abs(params["a"] - 37) / 50 - abs(params["b"] - 0.8) + (0.5 if params["c"] == "y" else 0.0)
    return Evaluation(fitness=score)


def test_ga_converges_on_toy_problem():
    ga = GeneticAlgorithm(TOY_SPACE, toy_fitness, GAConfig(population_size=30, generations=30, seed=1))
    result = ga.run()
    assert result.best_params["c"] == "y"
    assert abs(result.best_params["a"] - 37) <= 3
    assert abs(result.best_params["b"] - 0.8) <= 0.1


def test_ga_best_so_far_is_monotonic_and_elitism_preserves_best():
    ga = GeneticAlgorithm(TOY_SPACE, toy_fitness, GAConfig(population_size=12, generations=10, elitism=1, seed=3))
    history = ga.run().history
    best = [h.best_so_far for h in history]
    assert best == sorted(best)
    # com elitismo, o melhor da geração nunca piora
    gen_best = [h.best_fitness for h in history]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(gen_best, gen_best[1:]))


def test_ga_is_reproducible_with_same_seed():
    cfg = GAConfig(population_size=10, generations=5, seed=7)
    r1 = GeneticAlgorithm(TOY_SPACE, toy_fitness, cfg).run()
    r2 = GeneticAlgorithm(TOY_SPACE, toy_fitness, cfg).run()
    assert r1.best_params == r2.best_params
    assert [h.best_fitness for h in r1.history] == [h.best_fitness for h in r2.history]


def test_ga_seeds_initial_population_and_caches_evaluations():
    calls = []

    def counting_fitness(params):
        calls.append(params)
        return toy_fitness(params)

    seed = {"a": 37, "b": 0.8, "c": "y"}
    ga = GeneticAlgorithm(TOY_SPACE, counting_fitness, GAConfig(population_size=8, generations=4, seed=0),
                          initial_params=[seed])
    result = ga.run()
    assert result.best_evaluation.fitness >= toy_fitness(TOY_SPACE.decode(TOY_SPACE.encode(seed))).fitness
    assert len(calls) == result.n_evaluations == len({params_key(p) for p in calls})


def test_early_stopping_stops_before_max_generations():
    ga = GeneticAlgorithm(TOY_SPACE, lambda p: Evaluation(1.0),
                          GAConfig(population_size=6, generations=50, early_stopping_generations=3))
    assert len(ga.run().history) == 4  # geração 0 + 3 sem melhora


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        GAConfig(selection="inexistente")
    with pytest.raises(ValueError):
        GAConfig(population_size=4, elitism=4)


def test_random_search_respects_budget_and_curve_is_monotonic():
    params, evaluation, curve = random_search(TOY_SPACE, toy_fitness, n_evaluations=25, seed=0)
    assert len(curve) == 25
    assert curve == sorted(curve)
    assert evaluation.fitness == curve[-1]
