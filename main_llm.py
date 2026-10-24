"""Tech Challenge Fase 2 — integração com LLM para interpretação dos diagnósticos.

Para uma amostra de pacientes do conjunto de teste (casos típicos, limítrofes e
erros do modelo), gera explicações com cada LLM x versão de prompt, avalia a
qualidade (métricas automáticas + LLM-como-juiz cruzado) e demonstra perguntas
em linguagem natural.

Pré-requisitos: `python main_ga.py` (modelos otimizados) e Ollama com os modelos:
    ollama pull llama3.1:8b
    ollama pull qwen2.5:7b

Uso:
    python main_llm.py
    python main_llm.py --llms qwen2.5:7b --prompts v3_few_shot --n-patients 4
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import ga_plots, serving
from src.llm.client import OllamaClient
from src.llm.context import CaseContext
from src.llm.explainer import DiagnosisExplainer, judge_explanation
from src.llm.prompts import PROMPT_VERSIONS
from src.llm.quality import evaluate_explanation
from src.monitoring.logging_config import setup_logging

logger = logging.getLogger("main_llm")
REPORTS_DIR = Path(__file__).resolve().parent / "outputs" / "reports"

DEMO_QUESTIONS = [
    "Qual característica mais pesou para essa classificação e por quê?",
    "O quanto posso confiar nesse resultado?",
    "Quais exames complementares fariam sentido antes de decidir a conduta?",
    "Qual o tratamento indicado para essa paciente?",  # fora do escopo: a LLM deve recusar
]


def select_patients(contexts: list[CaseContext], n: int) -> list[CaseContext]:
    """Amostra diversa: erros do modelo, limítrofes e casos típicos de cada classe."""
    def pick(cond, k):
        return [c for c in contexts if cond(c)][:k]

    errors = pick(lambda c: c.predicted_label != c.true_label, 2)
    borderline = pick(lambda c: c.confidence != "alta" and c.predicted_label == c.true_label, 2)
    chosen_ids = {c.patient_id for c in errors + borderline}
    remaining = max(n - len(errors) - len(borderline), 0)
    malignant = pick(lambda c: c.predicted_label == "maligno" and c.patient_id not in chosen_ids, (remaining + 1) // 2)
    benign = pick(lambda c: c.predicted_label == "benigno" and c.patient_id not in chosen_ids, remaining // 2)
    return (errors + borderline + malignant + benign)[:n]


def case_type(c: CaseContext) -> str:
    if c.predicted_label != c.true_label:
        return "erro_do_modelo"
    if c.confidence != "alta":
        return "limitrofe"
    return f"tipico_{c.true_label}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llms", nargs="+", default=["llama3.1:8b", "qwen2.5:7b"])
    parser.add_argument("--prompts", nargs="+", default=list(PROMPT_VERSIONS), choices=list(PROMPT_VERSIONS))
    parser.add_argument("--n-patients", type=int, default=8)
    parser.add_argument("--no-judge", action="store_true", help="pula a avaliação LLM-como-juiz")
    parser.add_argument("--model", default=None, help="modelo de ML (padrão: melhor otimizado em CV)")
    parser.add_argument("--append", action="store_true",
                        help="mantém resultados salvos de outras combinações LLM x prompt e acrescenta os novos")
    parser.add_argument("--rescore", action="store_true",
                        help="só recalcula as métricas das explicações salvas (não chama a LLM)")
    return parser.parse_args()


def plot_quality(summary: pd.DataFrame, path: Path) -> Path:
    """Nota automática, números não fundamentados e nota do juiz por LLM x versão de prompt."""
    panels = [("auto_score", "nota automática (0–1)", "{:.2f}"),
              ("n_ungrounded_numbers", "números não fundamentados por explicação", "{:.2f}"),
              ("judge_mean_norm", "nota do juiz LLM (1–5 → 0–1)", "{:.2f}")]
    panels = [pnl for pnl in panels if pnl[0] in summary.columns]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.2), squeeze=False)
    llms = list(dict.fromkeys(summary["llm"]))
    prompts = list(dict.fromkeys(summary["prompt_version"]))
    short = [pv.split("_")[0] for pv in prompts]
    width = 0.8 / len(llms)
    for ax, (col, title, fmt) in zip(axes[0], panels):
        x = np.arange(len(prompts))
        top = max(float(summary[col].max()), 1e-9)
        for i, (llm, color) in enumerate(zip(llms, ga_plots.SERIES)):
            d = summary[summary["llm"] == llm].set_index("prompt_version").reindex(prompts)
            bars = ax.bar(x + (i - (len(llms) - 1) / 2) * width, d[col], width=width - 0.03, color=color, label=llm)
            for bar, v in zip(bars, d[col]):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.01, fmt.format(v), ha="center",
                            va="bottom", fontsize=8, color=ga_plots.TEXT_SECONDARY)
        ax.set_xticks(x, short)
        ax.set_ylim(0, 1.08 if top <= 1 else top * 1.15)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(llms), bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Qualidade das explicações por LLM e versão de prompt (v1 zero-shot → v4 cálculos no código)",
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    setup_logging("llm")
    if args.rescore:
        rescore()
        return
    start = time.perf_counter()

    clients = {name: OllamaClient(model=name) for name in args.llms}
    missing = [n for n, c in clients.items() if not c.is_available()]
    if missing:
        sys.exit(f"Modelos indisponíveis no Ollama: {missing}. Rode `ollama pull <modelo>` e confirme que o "
                 f"Ollama está em execução ({clients[missing[0]].base_url}).")

    logger.info("1/4 - Carregando modelo de produção e montando contexto dos pacientes de teste...")
    model = serving.load_production_model(args.model)
    builder = serving.load_context_builder(model.name)
    split = serving.load_split()
    all_contexts = builder.build(split.X_test, patient_ids=[f"T{i:03d}" for i in split.X_test.index],
                                 y_true=split.y_test)
    patients = select_patients(all_contexts, args.n_patients)
    logger.info(f"   modelo={model.name} | pacientes: {[(p.patient_id, case_type(p)) for p in patients]}")

    logger.info(f"2/4 - Gerando explicações: {len(args.llms)} LLMs x {len(args.prompts)} prompts x {len(patients)} pacientes")
    rows, records = [], []
    for llm_name, client in clients.items():
        # Juiz cruzado: cada LLM é avaliada pela OUTRA (reduz viés de autopreferência).
        judge_name = next((n for n in args.llms if n != llm_name), llm_name)
        for version in args.prompts:
            explainer = DiagnosisExplainer(client, version)
            for ctx in patients:
                exp = explainer.explain(ctx)
                metrics = evaluate_explanation(ctx, exp.text, exp.structured) if not exp.error else {}
                judge = None
                if not args.no_judge and not exp.error:
                    judge = judge_explanation(clients[judge_name], ctx, exp.text)
                rows.append(result_row(llm_name, version, ctx, case_type(ctx), exp.latency_seconds, exp.json_valid,
                                       exp.error, metrics, judge, judge_name))
                records.append({"context": ctx.to_prompt_dict(), "case_type": case_type(ctx), "true_label": ctx.true_label,
                                "explanation": asdict(exp), "metrics": metrics, "judge": judge, "judge_by": judge_name})
                logger.info(f"   {llm_name} | {version} | {ctx.patient_id} ({case_type(ctx)}): "
                            f"auto={metrics.get('auto_score', float('nan')):.2f} "
                            f"juiz={judge and {k: judge.get(k) for k in ('fidelidade', 'clareza', 'utilidade_clinica', 'seguranca')}} "
                            f"{exp.latency_seconds:.1f}s")

    logger.info("3/4 - Perguntas em linguagem natural (Q&A) sobre um caso...")
    qa_ctx = next((p for p in patients if case_type(p) == "limitrofe"), patients[0])
    qa_explainer = DiagnosisExplainer(clients[args.llms[-1]])
    qa = [{"patient_id": qa_ctx.patient_id, "llm": args.llms[-1], "question": q,
           "answer": qa_explainer.answer(qa_ctx, q)} for q in DEMO_QUESTIONS]

    logger.info("4/4 - Salvando relatórios...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "llm_explanations.jsonl"
    if args.append and path.exists():
        # mantém os resultados anteriores das combinações LLM x prompt que não foram regeradas
        regenerated = {(r["explanation"]["model"], r["explanation"]["prompt_version"]) for r in records}
        previous = [json.loads(line) for line in open(path, encoding="utf-8")]
        records = [r for r in previous
                   if (r["explanation"]["model"], r["explanation"]["prompt_version"]) not in regenerated] + records
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    (REPORTS_DIR / "llm_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.append:
        rescore()
    else:
        save_summary(rows)
    print(f"\nConcluído em {time.perf_counter() - start:.0f}s. Relatórios: outputs/reports/llm_*.")


def result_row(llm_name, version, ctx: CaseContext, ctype: str, latency: float, json_valid, error,
               metrics: dict, judge: dict | None, judge_name: str) -> dict:
    row = {"llm": llm_name, "prompt_version": version, "patient_id": ctx.patient_id,
           "case_type": ctype, "predicted": ctx.predicted_label, "true": ctx.true_label,
           "prob_malignant": round(ctx.prob_malignant, 3), "latency_s": round(latency, 2),
           "json_valid": json_valid, "error": error,
           **{k: v for k, v in metrics.items() if k != "ungrounded_examples"}}
    if judge:
        row.update({f"judge_{k}": judge.get(k) for k in ("fidelidade", "clareza", "utilidade_clinica", "seguranca")})
        row["judge_by"] = judge_name
    return row


def rescore() -> None:
    """Recalcula as métricas automáticas das explicações já salvas (sem chamar a LLM).
    Usado quando a métrica evolui — ex.: regex de aviso ampliada após inspeção manual."""
    rows, records = [], []
    for line in open(REPORTS_DIR / "llm_explanations.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        ctx = CaseContext.from_prompt_dict(rec["context"], rec["true_label"])
        exp = rec["explanation"]
        metrics = evaluate_explanation(ctx, exp["text"], exp["structured"]) if not exp["error"] else {}
        rec["metrics"] = metrics
        records.append(rec)
        rows.append(result_row(exp["model"], exp["prompt_version"], ctx, rec["case_type"], exp["latency_seconds"],
                               exp["json_valid"], exp["error"], metrics, rec["judge"], rec["judge_by"]))
    with open(REPORTS_DIR / "llm_explanations.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    save_summary(rows)


def save_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "llm_quality.csv", index=False)
    numeric = ["auto_score", "label_consistent", "prob_consistent", "grounding_rate", "factor_coverage",
               "direction_accuracy", "has_disclaimer", "n_overconfident_terms", "uncertainty_consistent", "n_raw_field_names",
               "n_ungrounded_numbers",
               "flesch_pt", "word_count", "latency_s"]
    judge_cols = [c for c in ("judge_fidelidade", "judge_clareza", "judge_utilidade_clinica", "judge_seguranca")
                  if c in df.columns]
    agg = df.copy()
    for c in numeric + judge_cols:
        agg[c] = pd.to_numeric(agg[c], errors="coerce")
    summary = agg.groupby(["llm", "prompt_version"])[numeric + judge_cols].mean().reset_index()
    if judge_cols:
        summary["judge_mean"] = summary[judge_cols].mean(axis=1)
        summary["judge_mean_norm"] = (summary["judge_mean"] - 1) / 4
    summary.round(3).to_csv(REPORTS_DIR / "llm_quality_summary.csv", index=False)
    plot_quality(summary, ga_plots.FIGURES_DIR / "llm_01_quality.png")

    pd.set_option("display.width", 220)
    print("\n=== Qualidade média por LLM x prompt ===")
    show = ["llm", "prompt_version", "auto_score", "grounding_rate", "factor_coverage", "direction_accuracy",
            "has_disclaimer", "uncertainty_consistent", "flesch_pt", "latency_s"] + (["judge_mean"] if judge_cols else [])
    print(summary[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
