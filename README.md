# AnesAgent

Perioperative anesthesia multi-agent prototype with three exposed agents: Perception, Decision, and Evaluation.

## Core commands

```bash
pip install -r requirements-anesthesia.txt
pip install -r requirements-training.txt

python scripts/train_event_router.py \
  --input Anes_Dataset_Thoracic/datasets/gpt_vitaldb_miller_all.jsonl \
  --out runs/event_router

python scripts/run_multimodal_or_agent.py \
  --input Anes_Dataset_Thoracic/datasets/gpt_vitaldb_miller_all.jsonl \
  --index 2 \
  --config configs/multimodal_or_agent.json \
  --out results/multimodal_or_agent/report.json
```

Data, trained models, and run outputs are intentionally excluded from git.
