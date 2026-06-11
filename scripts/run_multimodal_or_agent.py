import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anesthesia_agent.io_utils import dump_json, load_config
from src.anesthesia_agent.multimodal_or_workflow import MultimodalORAnesthesiaWorkflow


def _load_jsonl_sample(path: Path, index: int = 0, caseid: str = "") -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if caseid and str(row.get("caseid")) != str(caseid):
                continue
            if not caseid and i != index:
                continue
            return row
    target = f"caseid={caseid}" if caseid else f"index={index}"
    raise ValueError(f"No sample found for {target} in {path}")


def _load_input(path: str, index: int, caseid: str) -> Dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return _load_jsonl_sample(p, index=index, caseid=caseid)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal OR anesthesia agent.")
    parser.add_argument(
        "--input",
        type=str,
        default="Anes_Dataset_Thoracic/datasets/gpt_vitaldb_miller_all.jsonl",
        help="JSONL dataset file or JSON snapshot/case file.",
    )
    parser.add_argument("--index", type=int, default=0, help="0-based JSONL sample index.")
    parser.add_argument("--caseid", type=str, default="", help="Select the first JSONL row with this caseid.")
    parser.add_argument("--config", type=str, default="configs/multimodal_or_agent.json")
    parser.add_argument("--out", type=str, default="results/multimodal_or_agent/report.json")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    row = _load_input(args.input, index=int(args.index), caseid=args.caseid)
    workflow = MultimodalORAnesthesiaWorkflow(config=cfg)
    state = workflow.invoke(row)
    report = state.get("final_report", {})
    report["audit_trail"] = state.get("audit_trail", [])
    dump_json(args.out, report)

    fused = report.get("perception", {}).get("fused_context", {})
    evaluation = report.get("evaluation", {})
    print(f"[ok] multimodal OR report saved: {args.out}")
    print(f"caseid={report.get('caseid')}")
    print(f"primary_event={fused.get('primary_event')} severity={fused.get('severity')}")
    print(f"decision_mode={report.get('decision_mode')}")
    print(f"evaluation_score={evaluation.get('overall_score')}")
    print(f"human_review_required={report.get('human_review_required')}")
    if report.get("safety_red_lines"):
        print("safety_red_lines:")
        for item in report["safety_red_lines"]:
            print(f"- {item}")


if __name__ == "__main__":
    main()


