"""Gera notebooks/demo_fase2.ipynb (depois executado com nbconvert --execute)."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Tech Challenge — Fase 2: demonstração

Otimização dos modelos de diagnóstico da Fase 1 com **algoritmo genético** e interpretação dos
resultados com **LLM**. Este notebook carrega os artefatos gerados por `main_ga.py`,
`main_ga_cnn.py` e `main_llm.py` e demonstra cada componente.

> Ferramenta de apoio à decisão — não substitui a avaliação médica.
""")
code("""
import os, sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
from IPython.display import Image, Markdown, display
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
REPORTS, FIGS = ROOT / "outputs/reports", ROOT / "outputs/figures"
""")

md("""
## 1. O algoritmo genético em miniatura

Cada indivíduo é um vetor de genes em [0, 1); cada gene é decodificado para um hiperparâmetro
(inteiro, real/log ou categórico). Abaixo, o cromossomo da Random Forest e a decodificação de um
indivíduo aleatório e da configuração original da Fase 1.
""")
code("""
from src.ga.search_spaces import SEARCH_SPACES, BASELINE_PARAMS
space = SEARCH_SPACES["random_forest"]
rng = np.random.default_rng(0)
genome = space.random_genome(rng)
display(pd.DataFrame({"gene": space.names, "u (genoma)": genome.round(3),
                      "valor decodificado": list(space.decode(genome).values())}))
print("Fase 1 codificada:", space.encode(BASELINE_PARAMS["random_forest"]).round(3))
""")
md("Uma execução curta e ao vivo do GA na Decision Tree (3 gerações, população 10):")
code("""
from src import data_loader, preprocessing, modeling
from src.ga.engine import GAConfig, GeneticAlgorithm
from src.ga.fitness import CVFitness

df, _ = preprocessing.clean_dataset(data_loader.load_dataset())
split = modeling.train_test_split_dataset(df)
fitness = CVFitness("decision_tree", split.X_train, split.y_train)
ga = GeneticAlgorithm(SEARCH_SPACES["decision_tree"], fitness, GAConfig(population_size=10, generations=3),
                      initial_params=[BASELINE_PARAMS["decision_tree"]], n_jobs=-1)
result = ga.run()
display(pd.DataFrame(result.history_records())[["generation", "best_fitness", "mean_fitness", "best_so_far", "unique_evaluations", "diversity"]].round(4))
print("Melhores hiperparâmetros:", result.best_params)
""")

md("""
## 2. Resultados dos experimentos (execução completa)

4 experimentos (cada um varia **um** fator em relação ao E1), 3 seeds cada, 15 gerações:

| Experimento | Configuração |
|---|---|
| E1_referencia | pop. 20, mutação 0.10, torneio k=3, crossover uniforme |
| E2_populacao_grande | pop. **40** |
| E3_mutacao_alta | mutação **0.30**, sigma 0.25 |
| E4_roleta_blend | seleção por **roleta** + crossover **BLX-α** |
""")
code("""
summary = pd.read_csv(REPORTS / "ga_experiments_summary.csv")
summary[["model", "experiment", "best_fitness_mean", "best_fitness_std", "baseline_cv_fitness",
         "random_search_fitness", "evaluations_mean", "elapsed_seconds_mean"]].round(4)
""")
code('display(Image(FIGS / "ga_01_convergence.png"))')
code('display(Image(FIGS / "ga_05_experiments_summary.png"))')
code('display(Image(FIGS / "ga_03_ga_vs_random_search.png"))')
code('display(Image(FIGS / "ga_06_diversity.png"))')

md("## 3. Original (Fase 1) vs. otimizado — conjunto de teste")
code("""
comp = pd.read_csv(REPORTS / "comparison_original_vs_optimized.csv")
comp[[c for c in comp.columns if "_ci_" not in c]].round(4)
""")
code('display(Image(FIGS / "ga_04_original_vs_optimized.png"))')
code("""
best = json.loads((REPORTS / "ga_best_params.json").read_text(encoding="utf-8"))
pd.DataFrame({m: {"experimento": b["experiment"], "fitness CV": round(b["cv_fitness"], 4),
                  "fitness CV original": round(b["baseline_cv_fitness"], 4),
                  "hiperparâmetros": b["params"]} for m, b in best.items()}).T
""")

md("""
### 3.1 Comparação robusta: validação cruzada aninhada

O teste tem só 114 pacientes. Na validação aninhada o GA roda dentro de cada fold externo (5 folds × 2
repetições) e original e otimizado são comparados, pareados, em pacientes que o GA nunca viu.
""")
code("""
nested = pd.read_csv(REPORTS / "nested_cv_summary.csv")
nested[["model", "metric", "original_mean", "otimizado_mean", "diff_mean", "wins", "ties", "losses", "wilcoxon_p"]].round(4)
""")
code('display(Image(FIGS / "ga_07_nested_cv_paired.png"))')

md("## 4. Extra — GA na CNN (ResNet18, transfer learning, GPU)")
code("""
cnn_path = REPORTS / "cnn_comparison_original_vs_optimized.csv"
if cnn_path.exists():
    cnn = pd.read_csv(cnn_path)
    display(cnn[[c for c in cnn.columns if "_ci_" not in c]].round(4))
    display(json.loads((REPORTS / "cnn_ga_best_params.json").read_text(encoding="utf-8"))["best_params"])
    display(Image(FIGS / "cnn_ga_02_convergence.png"))
    display(Image(FIGS / "cnn_ga_01_original_vs_optimized.png"))
else:
    print("Rode `python main_ga_cnn.py` para gerar os resultados da CNN.")
""")

md("""
## 5. LLM — do número ao insight

O **contexto do caso** é tudo que a LLM recebe: predição, probabilidade, confiança, os 5 fatores
SHAP mais influentes (em pontos de probabilidade) e onde cada valor do paciente se situa em relação
aos casos benignos e malignos.
""")
code("""
from src import serving
model = serving.load_production_model()
builder = serving.load_context_builder(model.name)
contexts = builder.build(split.X_test, patient_ids=[f"T{i:03d}" for i in split.X_test.index], y_true=split.y_test)
ctx = next(c for c in contexts if c.confidence != "alta")   # um caso limítrofe
print("Modelo de produção:", model.name)
print(json.dumps(ctx.to_prompt_dict(), ensure_ascii=False, indent=2))
""")
code("""
from src.llm.client import OllamaClient
from src.llm.explainer import DiagnosisExplainer
from src.llm.quality import evaluate_explanation

llm = OllamaClient(model="qwen2.5:7b")
if llm.is_available():
    exp = DiagnosisExplainer(llm, "v4_calculos_no_codigo").explain(ctx)
    display(Markdown(exp.text))
    q = evaluate_explanation(ctx, exp.text, exp.structured)
    print({k: q[k] for k in ("auto_score", "grounding_rate", "factor_coverage", "direction_accuracy", "has_disclaimer", "flesch_pt")})
    print(f"latência: {exp.latency_seconds:.1f}s")
else:
    print("Ollama indisponível — exibindo uma explicação salva por main_llm.py")
    rec = [json.loads(l) for l in open(REPORTS / "llm_explanations.jsonl", encoding="utf-8")]
    display(Markdown(next(r for r in rec if r["explanation"]["prompt_version"] == "v4_calculos_no_codigo")["explanation"]["text"]))
""")
md("### Perguntas em linguagem natural")
code("""
qa = json.loads((REPORTS / "llm_qa.json").read_text(encoding="utf-8"))
for item in qa:
    display(Markdown(f"**Pergunta:** {item['question']}\\n\\n**Resposta ({item['llm']}):** {item['answer']}"))
""")

md("## 6. Avaliação da qualidade das interpretações")
code("""
qs = pd.read_csv(REPORTS / "llm_quality_summary.csv")
cols = [c for c in ["llm", "prompt_version", "auto_score", "label_consistent", "prob_consistent", "grounding_rate",
                    "factor_coverage", "direction_accuracy", "has_disclaimer", "flags_uncertainty", "flesch_pt",
                    "latency_s", "judge_fidelidade", "judge_clareza", "judge_utilidade_clinica", "judge_seguranca",
                    "judge_mean"] if c in qs.columns]
qs[cols].round(3)
""")
code('display(Image(FIGS / "llm_01_quality.png"))')

md("### Escalabilidade automática (Kubernetes HPA, teste local com kind)")
code("""
display(pd.read_csv(REPORTS / "k8s_load_test.csv"))
display(Image(FIGS / "infra_01_hpa_scaling.png"))
""")

md("""
## 7. API

Com a API rodando (`uvicorn src.api.app:app` ou `docker compose up`), o mesmo fluxo é exposto via HTTP:
""")
code("""
import requests
try:
    patient = requests.get("http://localhost:8000/examples", params={"n": 1}, timeout=3).json()[0]
    pred = requests.post("http://localhost:8000/predict", json={"patients": [patient]}, timeout=10).json()
    print(pred)
except requests.RequestException:
    print("API não está rodando — veja o README (seção 4).")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                                 "language_info": {"name": "python"}})
out = ROOT / "notebooks" / "demo_fase2.ipynb"
nbf.write(nb, out)
print(out)
