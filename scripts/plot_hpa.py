"""Gráfico da linha do tempo do HPA (gerada por scripts/k8s_demo.sh / coleta manual).

    python scripts/plot_hpa.py outputs/reports/k8s_hpa_timeline.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import ga_plots  # noqa: E402  (estilo e paleta compartilhados)


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    axes[0].plot(df["tempo_s"], df["cpu_alvo"], color=ga_plots.SERIES[1])
    axes[0].axhline(60, color=ga_plots.NEUTRAL, linestyle="--", linewidth=1.2)
    axes[0].annotate("alvo do HPA (60%)", (df["tempo_s"].max(), 60), textcoords="offset points", xytext=(0, 4),
                     ha="right", fontsize=8, color=ga_plots.TEXT_SECONDARY)
    axes[0].set_ylabel("CPU média (% do request)")
    axes[0].set_title("Carga e utilização de CPU", loc="left", fontweight="bold")
    axes[1].step(df["tempo_s"], df["replicas"], where="post", color=ga_plots.SERIES[0], label="réplicas desejadas")
    axes[1].step(df["tempo_s"], df["pods_prontos"], where="post", color=ga_plots.SERIES[2], linestyle=":",
                 label="pods prontos")
    axes[1].set_ylabel("réplicas")
    axes[1].set_xlabel("tempo (s)")
    axes[1].set_title("Réplicas ajustadas pelo HPA", loc="left", fontweight="bold")
    axes[1].legend(loc="upper right")
    fig.suptitle("Kubernetes HPA: escalabilidade automática sob carga (kind, local)", x=0.02, ha="left")
    fig.tight_layout()
    out = ga_plots.FIGURES_DIR / "infra_01_hpa_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/reports/k8s_hpa_timeline.csv")
