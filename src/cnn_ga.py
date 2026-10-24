"""EXTRA: algoritmo genético para os hiperparâmetros da CNN de transfer learning (ResNet18).

Cada indivíduo = um treino completo da rede, então o custo por avaliação é ordens de
grandeza maior que nos modelos tabulares. Para caber no orçamento (GPU RTX 5070):
- as imagens são pré-carregadas uma vez (uint8, 224x224) e ficam em memória/GPU;
- aumento de dados (flips) feito na GPU;
- precisão mista (bfloat16) e channels_last;
- fitness com treino curto (poucas épocas) na validação agrupada por paciente;
- só a configuração vencedora é re-treinada com o orçamento completo e vai ao teste.

Atenção: nas imagens (CBIS-DDSM), label 1 = maligno — o inverso do dataset tabular.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src import cnn_data
from src.cnn_model import build_resnet18_transfer
from src.ga.engine import Evaluation
from src.ga.genes import CategoricalGene, FloatGene, SearchSpace

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "outputs" / "cache"
IMAGE_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

CNN_SEARCH_SPACE = SearchSpace([
    FloatGene("lr", 1e-5, 3e-3, log=True),
    FloatGene("weight_decay", 1e-6, 1e-2, log=True),
    FloatGene("dropout", 0.0, 0.6),
    # a partir de qual bloco a ResNet é re-treinada ("fc" = só a cabeça)
    CategoricalGene("unfreeze_from", ("layer2", "layer3", "layer4", "fc")),
    CategoricalGene("batch_size", (16, 32, 64)),
    # multiplicador do peso da classe maligna na loss (1.0 = balanceamento padrão)
    FloatGene("pos_weight_scale", 0.5, 2.0),
])

# Pesos do fitness da CNN — diferentes dos modelos tabulares (0.5 F1 / 0.3 recall / 0.2 AUC).
# Com a CNN fraca (AUC ~0.73), aqueles pesos premiavam o classificador degenerado que
# chama tudo de maligno (recall 1.0 -> fitness ~0.69, acima da config. original). O
# ROC-AUC não depende do limiar e vale 0.5 para esse classificador, então recebe o maior
# peso: o GA passa a buscar capacidade real de discriminação.
CNN_FITNESS_WEIGHTS = {"f1_malignant": 0.4, "recall_malignant": 0.1, "roc_auc": 0.5}

# Configuração usada na Fase 1 (main_cnn_transfer.py).
CNN_BASELINE_PARAMS = {"lr": 1e-4, "weight_decay": 1e-4, "dropout": 0.4, "unfreeze_from": "layer3",
                       "batch_size": 32, "pos_weight_scale": 1.0}


@dataclass
class ImageData:
    x: torch.Tensor   # uint8 (N, 1, H, W) — convertido para float/RGB só no batch
    y: torch.Tensor   # float (N,)


def _load_split(manifest: pd.DataFrame, name: str) -> ImageData:
    """Carrega (com cache em disco) as imagens redimensionadas de um split."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"cbis_{name}_{IMAGE_SIZE}.pt"
    if path.exists():
        cached = torch.load(path)
        if len(cached["y"]) == len(manifest):
            return ImageData(cached["x"], cached["y"])
    images = np.stack([
        np.asarray(Image.open(p).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR), dtype=np.uint8)
        for p in manifest["full_path"]
    ])
    data = ImageData(torch.from_numpy(images).unsqueeze(1), torch.tensor(manifest["label"].values, dtype=torch.float32))
    torch.save({"x": data.x, "y": data.y}, path)
    return data


def load_image_splits(random_state: int = 42) -> dict[str, ImageData]:
    """Mesmo particionamento da Fase 1: teste oficial + validação agrupada por paciente."""
    manifest, _ = cnn_data.build_manifest(verbose=False)
    train_full = manifest[manifest["split"] == "train"].reset_index(drop=True)
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)
    train, val = cnn_data.split_train_val_by_patient(train_full, val_size=0.15, random_state=random_state)
    return {"train": _load_split(train, "train"), "val": _load_split(val, "val"), "test": _load_split(test, "test")}


def _prepare(x_uint8: torch.Tensor, device: torch.device, augment: bool) -> torch.Tensor:
    x = x_uint8.to(device, non_blocking=True).float().div_(255).repeat(1, 3, 1, 1)
    if augment:
        flip_h = torch.rand(x.shape[0], device=device) < 0.5
        flip_v = torch.rand(x.shape[0], device=device) < 0.5
        x[flip_h] = x[flip_h].flip(3)
        x[flip_v] = x[flip_v].flip(2)
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def predict_proba(model: nn.Module, data: ImageData, device: torch.device, batch_size: int = 128) -> np.ndarray:
    model.eval()
    probs = []
    for i in range(0, len(data.y), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(_prepare(data.x[i:i + batch_size], device, augment=False)).view(-1)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs)


def image_metrics(y_true: np.ndarray, probs: np.ndarray, prefix: str = "") -> dict[str, float]:
    preds = (probs >= 0.5).astype(int)
    return {
        f"{prefix}accuracy": accuracy_score(y_true, preds),
        f"{prefix}precision_malignant": precision_score(y_true, preds, pos_label=1, zero_division=0),
        f"{prefix}recall_malignant": recall_score(y_true, preds, pos_label=1, zero_division=0),
        f"{prefix}f1_malignant": f1_score(y_true, preds, pos_label=1, zero_division=0),
        f"{prefix}roc_auc": roc_auc_score(y_true, probs),
    }


def composite_fitness(metrics: dict[str, float], prefix: str = "val_",
                      weights: dict[str, float] | None = None) -> float:
    weights = weights or CNN_FITNESS_WEIGHTS
    return sum(w * metrics[f"{prefix}{m}"] for m, w in weights.items())


def train_cnn(params: dict, data: dict[str, ImageData], device: torch.device, epochs: int,
              patience: int | None = None, seed: int = 42) -> tuple[nn.Module, list[dict]]:
    """Treina a ResNet18 e devolve o modelo no melhor checkpoint (menor val_loss) + histórico."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, val = data["train"], data["val"]
    model = build_resnet18_transfer(unfreeze_from=params["unfreeze_from"], dropout=params["dropout"])
    model = model.to(device).to(memory_format=torch.channels_last)

    n_pos = float(train.y.sum())
    pos_weight = (len(train.y) - n_pos) / max(n_pos, 1) * params["pos_weight_scale"]
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    val_criterion = nn.BCEWithLogitsLoss()  # sem peso: val_loss comparável entre indivíduos
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history, best_loss, best_state, stall = [], float("inf"), None, 0
    bs = int(params["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train.y), generator=generator)
        train_loss = 0.0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            if len(idx) < 2:  # BatchNorm não aceita batch de 1
                continue
            x = _prepare(train.x[idx], device, augment=True)
            y = train.y[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = criterion(model(x).view(-1).float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(idx)
        scheduler.step()

        probs = predict_proba(model, val, device)
        val_loss = float(val_criterion(torch.logit(torch.tensor(probs).clamp(1e-6, 1 - 1e-6)), val.y))
        metrics = image_metrics(val.y.numpy(), probs, prefix="val_")
        history.append({"epoch": epoch, "train_loss": train_loss / len(order), "val_loss": val_loss,
                        **metrics, "val_fitness": composite_fitness(metrics)})
        if val_loss < best_loss:
            best_loss, stall = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
            if patience is not None and stall >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


@dataclass
class CNNFitness:
    """Fitness = composto (CNN_FITNESS_WEIGHTS) na validação, no checkpoint de menor val_loss.

    `repeats` > 1 treina a mesma configuração com seeds diferentes e usa a MÉDIA: o
    treino de rede neural é ruidoso (inicialização, ordem dos batches), e com um único
    treino o GA tende a premiar indivíduos "sortudos" que o elitismo depois não deixa sair.
    """
    data: dict[str, ImageData]
    device: torch.device
    epochs: int = 5
    repeats: int = 1

    def __call__(self, params: dict) -> Evaluation:
        start = time.perf_counter()
        runs = []
        for r in range(self.repeats):
            _, history = train_cnn(params, self.data, self.device, epochs=self.epochs, seed=42 + r)
            runs.append(min(history, key=lambda h: h["val_loss"]))
        metrics = {k: float(np.mean([run[k] for run in runs])) for k in runs[0] if k.startswith("val_")}
        metrics["val_fitness_std"] = float(np.std([run["val_fitness"] for run in runs]))
        metrics["best_epoch"] = float(np.mean([run["epoch"] for run in runs]))
        metrics["train_seconds"] = time.perf_counter() - start
        logger.info(f"   CNN {params} -> fitness={metrics['val_fitness']:.4f} "
                    f"(±{metrics['val_fitness_std']:.4f}, {metrics['train_seconds']:.0f}s)",
                    extra={"event": "cnn_eval", **metrics})
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return Evaluation(fitness=metrics["val_fitness"], metrics=metrics)
