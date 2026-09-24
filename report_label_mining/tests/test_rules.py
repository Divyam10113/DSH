import pandas as pd
import pytest

from knee_labels.constants import LABELS
from knee_labels.rules import extract_report, rule_features, rule_scores, score_counts

CASES = [
    # English
    ("Complete tear of the anterior cruciate ligament. Medial meniscus intact. Small joint effusion.",
     {"ACL": 1, "Medial Meniscus": 0, "Effusion": 1}),
    ("No ACL tear, no fracture.", {"ACL": 0, "Fracture": 0}),
    ("ACL tear, PCL intact.", {"ACL": 1}),
    ("ACL with no evidence of tear.", {"ACL": 0}),
    ("Medial meniscus tear, no lateral meniscus tear.", {"Medial Meniscus": 1, "Lateral Meniscus": 0}),
    ("Tear of the posterior horn of the medial meniscus.", {"Medial Meniscus": 1, "Lateral Meniscus": 0}),
    ("Horizontal tear of the lateral meniscus and medial femoral condyle contusion.",
     {"Lateral Meniscus": 1, "Medial Meniscus": 0, "Contusion": 1}),
    ("Tears of the medial and lateral menisci.", {"Medial Meniscus": 1, "Lateral Meniscus": 1}),
    ("Possible MCL sprain.", {"MCL": 0.5}),
    ("ACL tear, possible MCL sprain.", {"ACL": 1, "MCL": 0.5}),
    ("Effusion: none.", {"Effusion": 0}),
    ("Tricompartmental osteoarthritis with small Baker's cyst.",
     {"Medial OA": 1, "Lateral OA": 1, "PF OA": 1, "Baker's": 1}),
    ("Chondromalacia patellae. Medial meniscus tear.", {"PF OA": 1, "Medial OA": 0, "Medial Meniscus": 1}),
    ("Medial compartment osteoarthritis with osteophytes.", {"Medial OA": 1, "Lateral OA": 0}),
    ("Status post arthroscopy. No fracture or bone bruise.", {"Medial OA": 0, "Fracture": 0, "Contusion": 0}),
    ("Synovitis. Segond fracture.", {"Synovitis": 1, "Fracture": 1}),
    # German
    ("Kein Gelenkerguss. V.a. Innenmeniskusriss. VKB-Ruptur.",
     {"Effusion": 0, "Medial Meniscus": 0.5, "ACL": 1}),
    ("Außenmeniskus: Lappenriss im Hinterhorn. Innenmeniskus unauffällig.",
     {"Lateral Meniscus": 1, "Medial Meniscus": 0}),
    # Spanish
    ("Rotura del ligamento cruzado anterior. Sin derrame articular. Edema óseo en cóndilo femoral lateral.",
     {"ACL": 1, "Effusion": 0, "Contusion": 1}),
    # French
    ("Déchirure du ménisque interne. Pas d'épanchement. Kyste poplité.",
     {"Medial Meniscus": 1, "Effusion": 0, "Baker's": 1}),
    # Italian
    ("Lesione del menisco mediale. Versamento articolare. Non fratture.",
     {"Medial Meniscus": 1, "Effusion": 1, "Fracture": 0}),
]


@pytest.mark.parametrize("report,expected", CASES, ids=[c[0][:40] for c in CASES])
def test_rule_cases(report, expected):
    counts = extract_report(report)
    assert {k: score_counts(counts[k]) for k in expected} == expected


def test_unmentioned_labels_score_zero():
    counts = extract_report("Normal knee MRI.")
    assert all(score_counts(c) == 0 for c in counts.values())


def test_frame_shapes_and_missing_reports():
    reports = pd.Series(["ACL tear.", None, ""], index=[10, 11, 12])
    scores = rule_scores(reports.fillna(""))
    assert list(scores.columns) == LABELS
    assert list(scores.index) == [10, 11, 12]
    assert scores.loc[10, "ACL"] == 1 and scores.loc[11].sum() == 0

    feats = rule_features(reports.fillna(""))
    assert feats.shape == (3, 3 * len(LABELS))
    assert feats.loc[10, "ACL|affirmed"] == 1
