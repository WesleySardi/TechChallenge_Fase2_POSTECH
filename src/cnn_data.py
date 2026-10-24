"""Construção do manifesto de imagens (CBIS-DDSM) a partir dos CSVs de metadados.

O dataset CBIS-DDSM (versão "awsaf49/cbis-ddsm-breast-cancer-image-dataset" do Kaggle)
separa metadados (patient_id, lateralidade, view, diagnóstico) dos arquivos de imagem
reais. A ligação entre as duas fontes é feita via `dicom_info.csv`, cujo campo
`PatientID` (reaproveitado pelo autor do dataset) reproduz o identificador usado nos
CSVs de caso (`Mass-Training_P_00001_LEFT_CC_1`, por exemplo).

Uso as imagens já recortadas ("cropped images") na região da anormalidade, o que
transforma o problema em uma classificação binária direta por imagem, sem precisar lidar
com a mamografia completa + máscara de ROI.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Dataset (~6 GB) não é duplicado na Fase 2: usa CBIS_DDSM_DIR, ou data/archive local,
# ou a pasta da Fase 1 (../1_Fase/data/archive), nessa ordem.
ARCHIVE_DIR = next(
    (p for p in (
        Path(os.environ["CBIS_DDSM_DIR"]) if os.getenv("CBIS_DDSM_DIR") else None,
        _PROJECT_ROOT / "data" / "archive",
        _PROJECT_ROOT.parent / "1_Fase" / "data" / "archive",
    ) if p is not None and p.exists()),
    _PROJECT_ROOT / "data" / "archive",
)
CSV_DIR = ARCHIVE_DIR / "csv"

# label = 1 -> malignant (classe de interesse clínico), label = 0 -> benign.
# Observação: essa codificação é o INVERSO da usada no dataset estruturado da Fase 1
# (sklearn Breast Cancer Wisconsin usa 0 = malignant), documentado propositalmente
# aqui e no relatório técnico para evitar confusão entre os dois pipelines.
PATHOLOGY_TO_LABEL = {
    "MALIGNANT": 1,
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
}

CASE_FILES = [
    ("mass_case_description_train_set.csv", "Mass", "Training", "train"),
    ("mass_case_description_test_set.csv", "Mass", "Test", "test"),
    ("calc_case_description_train_set.csv", "Calc", "Training", "train"),
    ("calc_case_description_test_set.csv", "Calc", "Test", "test"),
]


@dataclass
class ManifestReport:
    n_rows_total: int
    n_missing_join: int
    n_missing_on_disk: int
    n_patients_overlap_removed: int
    n_train_rows_removed_for_overlap: int
    n_final: int


def _density_column(df: pd.DataFrame) -> str:
    return "breast_density" if "breast_density" in df.columns else "breast density"


def _load_cropped_image_index() -> pd.DataFrame:
    dicom = pd.read_csv(CSV_DIR / "dicom_info.csv")
    cropped = dicom[dicom["SeriesDescription"] == "cropped images"].copy()
    cropped["key"] = cropped["PatientID"].str.strip()
    cropped = cropped.drop_duplicates("key")
    return cropped[["key", "image_path"]]


def build_manifest(verbose: bool = True) -> tuple[pd.DataFrame, ManifestReport]:
    """Monta o DataFrame [image_path, label, pathology, abnormality_type, split, patient_id].

    `patient_id` é preservado explicitamente para permitir splits agrupados por paciente
    (um mesmo paciente pode ter várias imagens — diferentes views/achados — e misturá-las
    entre treino e validação causaria vazamento de dados)."""
    cropped_index = _load_cropped_image_index()

    parts = []
    for filename, prefix, split_label, split_name in CASE_FILES:
        df = pd.read_csv(CSV_DIR / filename)
        df["key"] = (
            f"{prefix}-{split_label}_" + df["patient_id"] + "_"
            + df["left or right breast"] + "_" + df["image view"]
            + "_" + df["abnormality id"].astype(str)
        )
        df["abnormality_type"] = prefix.lower()
        df["split"] = split_name
        parts.append(df[["key", "patient_id", "pathology", "abnormality_type", "split"]])

    manifest = pd.concat(parts, ignore_index=True)
    n_total = len(manifest)

    manifest = manifest.merge(cropped_index, on="key", how="left")
    n_missing_join = manifest["image_path"].isna().sum()
    manifest = manifest.dropna(subset=["image_path"]).copy()

    manifest["image_path"] = manifest["image_path"].str.replace("CBIS-DDSM/", "", n=1, regex=False)
    manifest["full_path"] = manifest["image_path"].apply(lambda p: ARCHIVE_DIR / p)

    exists_mask = manifest["full_path"].apply(lambda p: p.exists())
    n_missing_disk = (~exists_mask).sum()
    manifest = manifest[exists_mask].copy()

    manifest["label"] = manifest["pathology"].map(PATHOLOGY_TO_LABEL)
    manifest = manifest.dropna(subset=["label"]).copy()
    manifest["label"] = manifest["label"].astype(int)
    manifest = manifest.reset_index(drop=True)

    # O split oficial train/test do CBIS-DDSM NÃO é totalmente disjunto por paciente
    # (característica conhecida do dataset, não um bug de construção do manifesto):
    # ~31 pacientes aparecem tanto no "Training" quanto no "Test" set originais. Um
    # paciente presente nos dois lados vazaria informação (mesmas características de
    # tecido/densidade) do treino para o teste, inflando artificialmente as métricas.
    # Removemos as imagens desses pacientes do TREINO, preservando o teste oficial
    # intacto (é o benchmark fixo usado para reportar resultados).
    train_patients = set(manifest.loc[manifest["split"] == "train", "patient_id"])
    test_patients = set(manifest.loc[manifest["split"] == "test", "patient_id"])
    overlap_patients = train_patients & test_patients

    n_before_overlap_fix = len(manifest)
    manifest = manifest[
        ~((manifest["split"] == "train") & (manifest["patient_id"].isin(overlap_patients)))
    ].reset_index(drop=True)

    # Garante em tempo de execução o invariante que esta correção existe para impor:
    # nenhum patient_id pode aparecer em treino e teste simultaneamente. Se uma edição
    # futura neste bloco reintroduzir o vazamento, isso falha alto (erro), em vez de
    # inflar silenciosamente as métricas de teste.
    remaining_train_patients = set(manifest.loc[manifest["split"] == "train", "patient_id"])
    remaining_test_patients = set(manifest.loc[manifest["split"] == "test", "patient_id"])
    assert not (remaining_train_patients & remaining_test_patients), (
        "Vazamento de paciente entre treino/teste não foi eliminado corretamente"
    )

    report = ManifestReport(
        n_rows_total=n_total,
        n_missing_join=int(n_missing_join),
        n_missing_on_disk=int(n_missing_disk),
        n_patients_overlap_removed=len(overlap_patients),
        n_train_rows_removed_for_overlap=n_before_overlap_fix - len(manifest),
        n_final=len(manifest),
    )
    if verbose:
        print(f"Manifesto: {report.n_final}/{report.n_rows_total} imagens válidas "
              f"({report.n_missing_join} sem correspondência, {report.n_missing_on_disk} ausentes em disco)")
        print(f"   Removidas {report.n_train_rows_removed_for_overlap} imagens de treino de "
              f"{report.n_patients_overlap_removed} pacientes que também aparecem no teste oficial "
              f"(evita vazamento paciente treino->teste)")

    return manifest[["full_path", "label", "pathology", "abnormality_type", "split", "patient_id"]], report


def split_train_val_by_patient(
    train_full: pd.DataFrame, val_size: float = 0.15, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o conjunto de treino em treino/validação agrupando por `patient_id`.

    Um `train_test_split` comum poderia colocar imagens do mesmo paciente (diferentes
    views ou achados) em treino E validação simultaneamente, vazando informação e
    inflando as métricas de validação usadas para escolher checkpoint / early stopping.
    """
    n_splits = max(2, round(1 / val_size))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, val_idx = next(
        splitter.split(train_full, train_full["label"], groups=train_full["patient_id"])
    )
    train_split = train_full.iloc[train_idx].reset_index(drop=True)
    val_split = train_full.iloc[val_idx].reset_index(drop=True)

    overlap = set(train_split["patient_id"]) & set(val_split["patient_id"])
    assert not overlap, f"Vazamento de paciente entre treino/validação: {overlap}"

    return train_split, val_split
