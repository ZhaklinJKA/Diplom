import json
import glob
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# =========================
# SETTINGS
# =========================
DATA_DIR = Path("/home/chupchik/Рабочий стол/fisrt_stepD/AIT Alert Data Set/8263181/ait_ads")
OUT_DIR = Path("/home/chupchik/Рабочий стол/fisrt_stepD/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLE = 300_000           # сколько строк берем для ML-эксперимента
TEST_SIZE = 0.2
RANDOM_STATE = 42

TFIDF_MAX_FEATURES = 200_000
TFIDF_MIN_DF = 3
TFIDF_NGRAM_RANGE = (1, 2)

LR_MAX_ITER = 300


# =========================
# HELPERS: JSON PARSING
# =========================
def _get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def parse_jsonl_or_json(fp: Path) -> List[Dict[str, Any]]:
    """Support JSONL (1 record per line) and JSON (list/object)."""
    records: List[Dict[str, Any]] = []
    with fp.open("r", encoding="utf-8", errors="replace") as f:
        first = f.read(1)
        f.seek(0)

        # JSONL
        if first != "[":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except json.JSONDecodeError:
                    continue
            return records

        # JSON list/object
        try:
            data = json.load(f)
            if isinstance(data, list):
                records.extend([x for x in data if isinstance(x, dict)])
            elif isinstance(data, dict):
                records.append(data)
        except json.JSONDecodeError:
            pass

    return records


# =========================
# LOAD WAZUH
# =========================
def load_wazuh_dataset(data_dir: Path) -> pd.DataFrame:
    files = sorted(Path(p) for p in glob.glob(str(data_dir / "*wazuh*.json")))
    if not files:
        raise FileNotFoundError("Не нашёл *wazuh*.json в DATA_DIR")

    print("Found Wazuh files:", len(files))
    print("Files:", [f.name for f in files])

    rows = []
    for fp in files:
        recs = parse_jsonl_or_json(fp)
        for r in recs:
            ts = r.get("@timestamp") or _get(r, "timestamp")
            rows.append({
                "file": fp.name,
                "timestamp": ts,
                "location": _get(r, "location"),

                "agent_id": _get(r, "agent.id"),
                "agent_name": _get(r, "agent.name"),
                "agent_ip": _get(r, "agent.ip"),

                "hostname": _get(r, "predecoder.hostname"),
                "program": _get(r, "predecoder.program_name"),

                "full_log": _get(r, "full_log"),

                "rule_id": _get(r, "rule.id"),
                "rule_description": _get(r, "rule.description"),
                "rule_groups": _get(r, "rule.groups"),
                "rule_level": _get(r, "rule.level"),
            })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["rule_level"] = pd.to_numeric(df["rule_level"], errors="coerce")

    def groups_to_str(x):
        if isinstance(x, list):
            return ",".join(map(str, x))
        return str(x) if x is not None else None

    df["rule_groups_str"] = df["rule_groups"].apply(groups_to_str)
    return df


def basic_report(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=== BASIC INFO ===")
    lines.append(f"rows: {len(df)}")
    lines.append(f"columns: {list(df.columns)}")

    lines.append("\n=== MISSING (%) ===")
    miss = (df.isna().mean() * 100).sort_values(ascending=False).round(2)
    lines.append(miss.to_string())

    lines.append("\n=== timestamp range ===")
    lines.append(f"min: {df['timestamp'].min()}")
    lines.append(f"max: {df['timestamp'].max()}")

    lines.append("\n=== rule_level distribution ===")
    lines.append(df["rule_level"].value_counts(dropna=False).sort_index().to_string())

    lines.append("\n=== top rule_groups_str (top 10) ===")
    lines.append(df["rule_groups_str"].value_counts(dropna=False).head(10).to_string())

    report = "\n".join(lines)
    print(report)
    return report


# =========================
# TARGETS
# =========================
def main_group(groups_str: str):
    if not isinstance(groups_str, str) or not groups_str.strip():
        return None
    return groups_str.split(",")[0].strip()


def map_priority(level):
    if pd.isna(level):
        return None
    level = float(level)
    if level <= 3:
        return "low"
    elif level <= 6:
        return "medium"
    elif level <= 10:
        return "high"
    else:
        return "critical"


# =========================
# MODEL + INTERPRETATION
# =========================
def make_priority_pipeline(class_weight=None) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            max_features=TFIDF_MAX_FEATURES,
        )),
        ("clf", LogisticRegression(
            max_iter=LR_MAX_ITER,
            class_weight=class_weight
        ))
    ])


def print_and_save(text: str, path: Path):
    path.write_text(text, encoding="utf-8")
    print(f"\nSaved report: {path}")


def top_ngrams_by_class(pipeline: Pipeline, top_k: int = 15) -> str:
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    feature_names = tfidf.get_feature_names_out()
    coef = clf.coef_
    classes = clf.classes_

    out_lines = []
    for i, label in enumerate(classes):
        top_pos = np.argsort(coef[i])[-top_k:][::-1]
        out_lines.append(f"\nTop ngrams for PRIORITY='{label}':")
        out_lines.append(", ".join(feature_names[j] for j in top_pos))
    return "\n".join(out_lines)


def class_weights_for_y(y_train: np.ndarray) -> str:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    pairs = sorted(zip(classes, weights), key=lambda x: x[1], reverse=True)

    lines = ["Priority class weights (balanced):"]
    for c, w in pairs:
        lines.append(f"{c:>8} -> {w:.3f}")
    return "\n".join(lines)


# =========================
# MAIN
# =========================
def main():
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Load
    print("DATA_DIR:", DATA_DIR)
    print("OUT_DIR:", OUT_DIR)
    df_wazuh = load_wazuh_dataset(DATA_DIR)

    # 2) Report + Save data artifacts
    rep = basic_report(df_wazuh)
    print_and_save(rep, OUT_DIR / f"report_basic_{ts_tag}.txt")

    parquet_path = OUT_DIR / "ait_ads_wazuh.parquet"
    df_wazuh.to_parquet(parquet_path, index=False)
    print("\nSaved parquet:", parquet_path)

    sample_csv_path = OUT_DIR / "ait_ads_wazuh_sample_50k.csv"
    df_wazuh.sample(n=min(50_000, len(df_wazuh)), random_state=RANDOM_STATE).to_csv(sample_csv_path, index=False)
    print("Saved sample csv:", sample_csv_path)

    # 3) Targets
    df = df_wazuh.copy()
    df["y_type"] = df["rule_groups_str"].apply(main_group)
    df["y_priority"] = df["rule_level"].apply(map_priority)

    # Keep only usable rows for ML
    df_ml = df.dropna(subset=["full_log", "y_type", "y_priority"]).copy()
    print("\nAfter dropna:", len(df_ml))

    # Print distributions (for diploma)
    type_dist = df_ml["y_type"].value_counts()
    prio_dist = df_ml["y_priority"].value_counts()

    dist_text = []
    dist_text.append("=== Distributions after dropna ===")
    dist_text.append("\nType distribution (y_type):")
    dist_text.append(type_dist.to_string())
    dist_text.append("\nPriority distribution (y_priority):")
    dist_text.append(prio_dist.to_string())
    dist_report = "\n".join(dist_text)
    print(dist_report)
    print_and_save(dist_report, OUT_DIR / f"report_distributions_{ts_tag}.txt")

    # 4) Sample for experiments
    df_s = df_ml.sample(n=min(N_SAMPLE, len(df_ml)), random_state=RANDOM_STATE).copy()
    print("\nSample size:", len(df_s))

    X = df_s["full_log"].astype(str).values
    y = df_s["y_priority"].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("\nTrain size:", len(X_train), "Test size:", len(X_test))

    # 5) PRIORITY baseline (no balancing)
    baseline = make_priority_pipeline(class_weight=None)
    baseline.fit(X_train, y_train)
    pred_base = baseline.predict(X_test)

    base_report = classification_report(y_test, pred_base, zero_division=0)
    base_cm = confusion_matrix(y_test, pred_base)

    base_text = []
    base_text.append("=== PRIORITY BASELINE (no class_weight) ===")
    base_text.append(base_report)
    base_text.append("\nConfusion matrix:\n" + str(base_cm))
    base_text = "\n".join(base_text)
    print("\n" + base_text)
    print_and_save(base_text, OUT_DIR / f"report_priority_baseline_{ts_tag}.txt")

    # 6) PRIORITY balanced
    balanced = make_priority_pipeline(class_weight="balanced")
    balanced.fit(X_train, y_train)
    pred_bal = balanced.predict(X_test)

    bal_report = classification_report(y_test, pred_bal, zero_division=0)
    bal_cm = confusion_matrix(y_test, pred_bal)

    bal_text = []
    bal_text.append("=== PRIORITY BALANCED (class_weight='balanced') ===")
    bal_text.append(bal_report)
    bal_text.append("\nConfusion matrix:\n" + str(bal_cm))
    bal_text = "\n".join(bal_text)
    print("\n" + bal_text)
    print_and_save(bal_text, OUT_DIR / f"report_priority_balanced_{ts_tag}.txt")

    # 7) Explain: class weights + top ngrams
    cw_text = class_weights_for_y(y_train)
    tn_text = top_ngrams_by_class(balanced, top_k=15)

    explain_text = "\n".join([
        "=== PRIORITY INTERPRETATION (balanced model) ===",
        cw_text,
        "\n",
        tn_text
    ])
    print("\n" + explain_text)
    print_and_save(explain_text, OUT_DIR / f"report_priority_interpretation_{ts_tag}.txt")

    print("\nDone. All reports are saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
