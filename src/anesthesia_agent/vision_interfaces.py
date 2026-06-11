"""Vision/OCR/VLM adapters for the anesthesia Perception Agent.

The adapters define the interface first:
- Fast Stream can later be replaced by real monitor OCR or a lightweight visual
  encoder.
- Slow Stream can later call Qwen-VL or another OpenAI-compatible VLM.

For the current thoracic anesthesia dataset, ``monitor_snapshot_ocr`` uses the
structured snapshot as a stand-in for monitor OCR, so the full LangGraph workflow
can run before a visual model is trained or deployed.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


class VisionAdapterError(RuntimeError):
    pass


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clinical_assessment(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    obj = snapshot.get("clinical_assessment", {})
    return obj if isinstance(obj, dict) else {}


def _recent(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    obj = _clinical_assessment(snapshot).get("recent_state_mean", {})
    return obj if isinstance(obj, dict) else {}


def _alarm_tags(snapshot: Dict[str, Any]) -> List[str]:
    tags = _clinical_assessment(snapshot).get("alarm_tags", [])
    return [str(x) for x in tags] if isinstance(tags, list) else []


def monitor_snapshot_ocr(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Fast Stream placeholder: structured snapshot as monitor OCR output."""
    recent = _recent(snapshot)
    return {
        "adapter": "monitor_snapshot_ocr",
        "modality": "structured_snapshot",
        "extracted_vitals": {
            "HR_bpm": _as_float(recent.get("HR_bpm")),
            "MAP_mmhg": _as_float(recent.get("MAP_mmhg")),
            "SBP_mmhg": _as_float(recent.get("SBP_mmhg")),
            "DBP_mmhg": _as_float(recent.get("DBP_mmhg")),
            "SpO2_pct": _as_float(recent.get("SpO2_pct")),
            "EtCO2_mmhg": _as_float(recent.get("EtCO2_mmhg")),
            "BIS": _as_float(recent.get("BIS")),
            "SVV_pct": _as_float(recent.get("SVV_pct")),
        },
        "alarm_tags": _alarm_tags(snapshot),
        "confidence": "high_for_structured_dataset",
        "note": "Dataset-backed OCR interface. Replace with real monitor OCR later.",
    }


def _image_to_data_url(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise VisionAdapterError(f"image not found: {path}")
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def openai_compatible_vlm_describe(
    image_path: str,
    config: Dict[str, Any],
    prompt: str = "",
) -> Dict[str, Any]:
    """Slow Stream adapter for Qwen-VL or any OpenAI-compatible VLM endpoint."""
    api_base = str(config.get("vlm_api_base") or config.get("llm_api_base") or "http://127.0.0.1:8000/v1")
    api_base = api_base.rstrip("/")
    model = str(config.get("vlm_model") or config.get("llm_model") or "Qwen-VL")
    timeout_s = int(config.get("vlm_timeout_s", config.get("llm_timeout_s", 60)))
    data_url = _image_to_data_url(image_path)
    user_prompt = prompt or (
        "请作为麻醉感知智能体分析这张监护仪/波形/TEE/手术视野图像。"
        "输出JSON，字段包括：image_type, visible_values, waveform_abnormalities, "
        "clinical_description, uncertainty。不要输出内部推理过程。"
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是麻醉临床多模态感知智能体，只做图像描述和结构化提取，不给自动医嘱。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": float(config.get("vlm_temperature", 0.1)),
        "max_tokens": int(config.get("vlm_max_tokens", 600)),
    }
    req = urllib.request.Request(
        api_base + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    content = str(obj["choices"][0]["message"]["content"]).strip()
    return {
        "adapter": "openai_compatible_vlm",
        "model": model,
        "image_path": image_path,
        "raw_output": content,
    }


def run_slow_vision_adapter(snapshot: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Run optional VLM path if enabled and an image exists."""
    if not bool(config.get("use_vlm", False)):
        return {
            "adapter": "vlm_disabled",
            "enabled": False,
            "note": "Set use_vlm=true and vlm_model/vlm_api_base to enable Qwen-VL.",
        }

    image_path = str(snapshot.get("waveform_image_path") or "")
    if not image_path:
        return {
            "adapter": "openai_compatible_vlm",
            "enabled": True,
            "error": "snapshot has no waveform_image_path",
        }

    if not Path(image_path).exists() and Path("Anes_Dataset_Thoracic").exists():
        candidate = Path("Anes_Dataset_Thoracic") / image_path
        if candidate.exists():
            image_path = str(candidate)

    try:
        result = openai_compatible_vlm_describe(image_path, config)
        result["enabled"] = True
        return result
    except Exception as exc:
        return {
            "adapter": "openai_compatible_vlm",
            "enabled": True,
            "image_path": image_path,
            "error": str(exc),
        }
