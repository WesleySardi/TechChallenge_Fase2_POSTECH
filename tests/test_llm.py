import json

import pytest

from src.llm.client import FakeLLMClient, LLMError
from src.llm.context import CaseContextBuilder, confidence_level, feature_name_pt
from src.llm.explainer import DiagnosisExplainer, parse_json, render_markdown
from src.llm.prompts import EXPLANATION_SCHEMA, PROMPT_VERSIONS, build_explanation_messages
from src.llm.quality import evaluate_explanation, extract_numbers, flesch_pt


@pytest.fixture(scope="module")
def contexts(trained_pipeline, split):
    builder = CaseContextBuilder(trained_pipeline, "logistic_regression", split.X_train, split.y_train,
                                 model_performance={"test_recall_malignant": 0.95}, top_k=4, background_size=30)
    return builder.build(split.X_test.iloc[:6], y_true=split.y_test.iloc[:6])


def good_answer(ctx) -> dict:
    """Resposta 'perfeita' montada a partir do próprio contexto."""
    return {
        "resumo": f"O modelo classificou a lesão como {ctx.predicted_label}."
                  + ("" if ctx.confidence == "alta" else " Caso limítrofe, com cautela."),
        "classificacao_modelo": ctx.predicted_label,
        "probabilidade_malignidade": round(ctx.prob_malignant, 3),
        "nivel_confianca": ctx.confidence,
        "fatores_principais": [
            {"caracteristica": f.feature_pt, "valor": round(f.value, 3), "efeito": f.effect,
             "interpretacao": f"Percentil {round(f.percentile)}."}
            for f in ctx.top_factors
        ],
        "insights_acionaveis": ["Correlacionar com exame de imagem."],
        "limitacoes": ["Considera apenas morfologia nuclear."],
        "aviso": "Ferramenta de apoio à decisão; não substitui a avaliação médica.",
    }


# ------------------------------------------------------------------ contexto

def test_feature_name_translation():
    assert feature_name_pt("worst concave points").startswith("pontos côncavos")
    assert "erro padrão" in feature_name_pt("radius error")
    assert "média" in feature_name_pt("mean area")


@pytest.mark.parametrize("prob, expected", [(0.97, "alta"), (0.03, "alta"), (0.75, "moderada"), (0.55, "baixa")])
def test_confidence_level(prob, expected):
    assert confidence_level(prob) == expected


def test_context_is_consistent_with_model(contexts, trained_pipeline, split):
    probs = trained_pipeline.predict_proba(split.X_test.iloc[:6])[:, 0]  # classe 0 = maligno
    for ctx, p in zip(contexts, probs):
        assert ctx.prob_malignant == pytest.approx(p)
        assert ctx.predicted_label == ("maligno" if p >= 0.5 else "benigno")
        assert len(ctx.top_factors) == 4
        contributions = [abs(f.contribution) for f in ctx.top_factors]
        assert contributions == sorted(contributions, reverse=True)
        for f in ctx.top_factors:
            assert f.effect == ("aumenta_risco" if f.contribution > 0 else "reduz_risco")
            assert 0 <= f.percentile <= 100


def test_prompt_dict_hides_true_label(contexts):
    d = contexts[0].to_prompt_dict()
    assert "true_label" not in d
    assert contexts[0].true_label in {"maligno", "benigno"}


# ------------------------------------------------------------------ prompts

@pytest.mark.parametrize("version", list(PROMPT_VERSIONS))
def test_prompt_versions_build_messages(contexts, version):
    messages = build_explanation_messages(contexts[0], version)
    assert messages[-1]["role"] == "user"
    assert str(round(contexts[0].prob_malignant, 3)) in messages[-1]["content"]
    if version == "v3_few_shot":
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    if version == "v1_zero_shot":
        assert len(messages) == 1


# ------------------------------------------------------------------ explainer

def test_explainer_parses_structured_output_and_sends_schema(contexts):
    ctx = contexts[0]
    client = FakeLLMClient(response=good_answer(ctx))
    exp = DiagnosisExplainer(client, "v3_few_shot").explain(ctx)
    assert exp.json_valid
    assert exp.structured["classificacao_modelo"] == ctx.predicted_label
    assert client.calls[0]["json_schema"] == EXPLANATION_SCHEMA
    assert "Fatores principais" in exp.text


def test_explainer_handles_llm_failure(contexts):
    class Broken(FakeLLMClient):
        def chat(self, *a, **k):
            raise LLMError("offline")
    exp = DiagnosisExplainer(Broken(), "v2_estruturado").explain(contexts[0])
    assert exp.error == "offline" and exp.text == ""


def test_parse_json_accepts_fenced_block():
    assert parse_json('Aqui está:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json("sem json") is None


def test_render_markdown_contains_key_fields(contexts):
    md = render_markdown(good_answer(contexts[0]))
    assert contexts[0].predicted_label in md and "apoio à decisão" in md


# ------------------------------------------------------------------ qualidade

def test_quality_perfect_answer_scores_high(contexts):
    ctx = contexts[0]
    answer = good_answer(ctx)
    metrics = evaluate_explanation(ctx, render_markdown(answer), answer)
    assert metrics["label_consistent"] and metrics["prob_consistent"]
    assert metrics["direction_accuracy"] == 1.0
    assert metrics["factor_coverage"] == 1.0
    assert metrics["has_disclaimer"]
    assert metrics["grounding_rate"] == 1.0, metrics["ungrounded_examples"]


def test_quality_detects_hallucination_and_wrong_label(contexts):
    ctx = contexts[0]
    answer = good_answer(ctx)
    answer["classificacao_modelo"] = "benigno" if ctx.predicted_label == "maligno" else "maligno"
    answer["fatores_principais"][0]["efeito"] = "reduz_risco" if ctx.top_factors[0].effect == "aumenta_risco" else "aumenta_risco"
    answer["resumo"] += " A paciente tem 63.7 anos e tumor de 4.83 cm. Certamente é câncer."
    answer["aviso"] = ""
    metrics = evaluate_explanation(ctx, render_markdown(answer), answer)
    assert not metrics["label_consistent"]
    assert metrics["direction_accuracy"] < 1.0
    assert metrics["n_ungrounded_numbers"] >= 2
    assert metrics["n_overconfident_terms"] >= 1
    assert metrics["auto_score"] < evaluate_explanation(ctx, render_markdown(good_answer(ctx)), good_answer(ctx))["auto_score"]


def test_extract_numbers_handles_decimal_comma():
    assert extract_numbers("probabilidade de 71,5% e valor 0.162") == [71.5, 0.162]


def test_flesch_simple_text_is_easier_than_complex():
    simple = "O caso é benigno. O risco é baixo. Não há sinais graves."
    complex_ = ("A heterogeneidade morfométrica nuclear caracterizada pela concavidade acentuada "
                "evidencia irregularidades topológicas compatíveis com proliferação neoplásica.")
    assert flesch_pt(simple) > flesch_pt(complex_)


def test_schema_is_valid_json():
    json.dumps(EXPLANATION_SCHEMA)


def test_quality_penalizes_raw_json_field_names(contexts):
    ctx = contexts[0]
    parrot = (f"predicted_label: {ctx.predicted_label}. prob_malignant: {round(ctx.prob_malignant, 3)}. "
              "Não substitui a avaliação médica.")
    metrics = evaluate_explanation(ctx, parrot)
    assert metrics["n_raw_field_names"] == 2
    assert metrics["auto_score"] < 1.0


@pytest.mark.parametrize("aviso", [
    "Não deve ser considerado como diagnóstico definitivo.",
    "O diagnóstico final é da responsabilidade do médico.",
    "Deve ser confirmado por exames adicionais, como histopatologia.",
    "Ferramenta de apoio à decisão; não substitui a avaliação médica.",
])
def test_disclaimer_variants_are_recognized(contexts, aviso):
    assert evaluate_explanation(contexts[0], f"Resumo do caso. {aviso}")["has_disclaimer"]


def test_borderline_claim_on_high_confidence_is_inconsistent(contexts):
    ctx = next((c for c in contexts if c.confidence == "alta"), None)
    if ctx is None:
        pytest.skip("nenhum caso de confiança alta na amostra")
    assert not evaluate_explanation(ctx, "Caso limítrofe. Não substitui a avaliação médica.")["uncertainty_consistent"]


def test_negated_borderline_is_not_a_contradiction(contexts):
    ctx = next((c for c in contexts if c.confidence == "alta"), None)
    if ctx is None:
        pytest.skip("nenhum caso de confiança alta na amostra")
    assert evaluate_explanation(ctx, "O caso não é limítrofe. Não substitui a avaliação médica.")["uncertainty_consistent"]
