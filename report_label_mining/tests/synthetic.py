"""Tiny synthetic stand-in for the competition's train.csv / test.csv, for tests and dry runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from knee_labels.constants import ID_COL, LABELS, REPORT_COL

# (positive phrase, negative phrase) per language and label.
PHRASES = {
    "en": {
        "ACL": ("Complete tear of the anterior cruciate ligament.", "ACL intact."),
        "MCL": ("MCL sprain.", "The medial collateral ligament is intact."),
        "Medial Meniscus": ("Tear of the posterior horn of the medial meniscus.", "Medial meniscus intact."),
        "Lateral Meniscus": ("Lateral meniscus tear.", "No lateral meniscus tear."),
        "Medial OA": ("Medial compartment osteoarthritis.", "Medial compartment cartilage preserved."),
        "Lateral OA": ("Lateral compartment osteoarthritis.", "Lateral compartment cartilage preserved."),
        "PF OA": ("Patellofemoral osteoarthritis.", "Patellofemoral cartilage preserved."),
        "Effusion": ("Moderate joint effusion.", "No joint effusion."),
        "Synovitis": ("Synovitis.", "No synovitis."),
        "Baker's": ("Baker's cyst.", "No Baker's cyst."),
        "Contusion": ("Bone bruise of the lateral femoral condyle.", "No bone bruise."),
        "Fracture": ("Fracture of the tibial plateau.", "No fracture."),
    },
    "de": {
        "ACL": ("Ruptur des vorderen Kreuzbandes.", "Vorderes Kreuzband intakt."),
        "MCL": ("Innenbandläsion.", "Innenband unauffällig."),
        "Medial Meniscus": ("Innenmeniskusriss im Hinterhorn.", "Innenmeniskus unauffällig."),
        "Lateral Meniscus": ("Außenmeniskusriss.", "Außenmeniskus unauffällig."),
        "Medial OA": ("Arthrose medial betont.", "Knorpel medial erhalten."),
        "Lateral OA": ("Arthrose lateral betont.", "Knorpel lateral erhalten."),
        "PF OA": ("Retropatellararthrose.", "Retropatellarer Knorpel erhalten."),
        "Effusion": ("Deutlicher Gelenkerguss.", "Kein Gelenkerguss."),
        "Synovitis": ("Synovitis.", "Keine Synovitis."),
        "Baker's": ("Baker-Zyste.", "Keine Baker-Zyste."),
        "Contusion": ("Knochenmarködem im lateralen Femurkondylus.", "Kein Knochenmarködem."),
        "Fracture": ("Tibiakopffraktur.", "Keine Fraktur."),
    },
    "es": {
        "ACL": ("Rotura del ligamento cruzado anterior.", "Ligamento cruzado anterior íntegro."),
        "MCL": ("Esguince del ligamento colateral medial.", "Ligamento colateral medial normal."),
        "Medial Meniscus": ("Rotura del menisco interno.", "Menisco interno sin alteraciones."),
        "Lateral Meniscus": ("Rotura del menisco externo.", "Menisco externo sin alteraciones."),
        "Medial OA": ("Artrosis del compartimento medial.", "Compartimento medial conservado."),
        "Lateral OA": ("Artrosis del compartimento lateral.", "Compartimento lateral conservado."),
        "PF OA": ("Artrosis femoropatelar.", "Cartílago femoropatelar conservado."),
        "Effusion": ("Derrame articular.", "Sin derrame articular."),
        "Synovitis": ("Sinovitis.", "Sin sinovitis."),
        "Baker's": ("Quiste de Baker.", "Sin quiste de Baker."),
        "Contusion": ("Edema óseo en cóndilo femoral lateral.", "Sin edema óseo."),
        "Fracture": ("Fractura de meseta tibial.", "Sin fracturas."),
    },
}
PREVALENCE = np.array([0.3, 0.2, 0.35, 0.25, 0.3, 0.15, 0.3, 0.45, 0.15, 0.2, 0.25, 0.08])


def make_dataset(n: int = 300, labeled_frac: float = 0.4, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    langs = rng.choice(list(PHRASES), size=n, p=[0.5, 0.3, 0.2])
    Y = (rng.random((n, len(LABELS))) < PREVALENCE).astype(float)
    reports = []
    for i in range(n):
        parts = [PHRASES[langs[i]][label][0 if Y[i, j] else 1] for j, label in enumerate(LABELS)]
        rng.shuffle(parts)
        reports.append(" ".join(parts))
    train = pd.DataFrame({ID_COL: [f"1.2.826.{i}" for i in range(n)], REPORT_COL: reports})
    labeled = rng.random(n) < labeled_frac
    for j, label in enumerate(LABELS):
        train[label] = np.where(labeled, Y[:, j], np.nan)
    test = pd.DataFrame({ID_COL: [f"9.9.9.{i}" for i in range(3)]})
    train.attrs["hidden_Y"] = Y  # true labels of all studies, for checking pseudo-label quality
    return train, test


def write_dataset(data_dir: Path, **kwargs) -> pd.DataFrame:
    train, test = make_dataset(**kwargs)
    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    return train
