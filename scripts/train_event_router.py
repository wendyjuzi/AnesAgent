import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "Training dependencies are missing. Install with: pip install -r requirements-training.txt\n"
        f"Import error: {exc}"
    )

from src.anesthesia_agent.trained_event_router import (
    derive_event_label,
    derive_severity_label,
    extract_features,
)
from src.anesthesia_agent.io_utils import dump_json


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def maybe_stratify(labels: List[str]):
    counts = Counter(labels)
    return labels if len(counts) > 1 and min(counts.values()) >= 2 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train event router models for the anesthesia agent.")
    parser.add_argument("--input", default="Anes_Dataset_Thoracic/datasets/gpt_vitaldb_miller_all.jsonl")
    parser.add_argument("--out", default="runs/event_router")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--n_estimators", type=int, default=200)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.max_rows > 0:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("No training rows loaded.")

    features = [extract_features(r) for r in rows]
    y_event = [derive_event_label(r) for r in rows]
    y_severity = [derive_severity_label(r) for r in rows]

    indices = list(range(len(rows)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=maybe_stratify(y_event),
    )

    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform([features[i] for i in train_idx])
    x_test = vectorizer.transform([features[i] for i in test_idx])

    event_model = RandomForestClassifier(
        n_estimators=int(args.n_estimators),
        random_state=int(args.seed),
        class_weight="balanced",
        n_jobs=-1,
    )
    severity_model = RandomForestClassifier(
        n_estimators=int(args.n_estimators),
        random_state=int(args.seed) + 1,
        class_weight="balanced",
        n_jobs=-1,
    )

    event_model.fit(x_train, [y_event[i] for i in train_idx])
    severity_model.fit(x_train, [y_severity[i] for i in train_idx])

    event_pred = event_model.predict(x_test)
    severity_pred = severity_model.predict(x_test)
    y_event_test = [y_event[i] for i in test_idx]
    y_severity_test = [y_severity[i] for i in test_idx]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "event_router.joblib"
    bundle = {
        "vectorizer": vectorizer,
        "event_model": event_model,
        "severity_model": severity_model,
        "metadata": {
            "input": args.input,
            "n_rows": len(rows),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "seed": int(args.seed),
            "label_source": "explicit labels if present, otherwise weak labels from alarms/vitals/sample_category",
        },
    }
    joblib.dump(bundle, model_path)

    metrics = {
        "model_path": str(model_path),
        "n_rows": len(rows),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "event_label_counts": dict(Counter(y_event)),
        "severity_label_counts": dict(Counter(y_severity)),
        "event_accuracy": float(accuracy_score(y_event_test, event_pred)),
        "severity_accuracy": float(accuracy_score(y_severity_test, severity_pred)),
        "event_report": classification_report(y_event_test, event_pred, zero_division=0, output_dict=True),
        "severity_report": classification_report(y_severity_test, severity_pred, zero_division=0, output_dict=True),
        "event_confusion_matrix": confusion_matrix(y_event_test, event_pred, labels=sorted(set(y_event))).tolist(),
        "event_confusion_labels": sorted(set(y_event)),
        "severity_confusion_matrix": confusion_matrix(y_severity_test, severity_pred, labels=sorted(set(y_severity))).tolist(),
        "severity_confusion_labels": sorted(set(y_severity)),
    }
    dump_json(str(out_dir / "metrics.json"), metrics)

    print(f"[ok] trained event router: {model_path}")
    print(f"rows={len(rows)} train={len(train_idx)} test={len(test_idx)}")
    print(f"event_accuracy={metrics['event_accuracy']:.4f}")
    print(f"severity_accuracy={metrics['severity_accuracy']:.4f}")


if __name__ == "__main__":
    main()
