"""Trainable event-router utilities for anesthesia snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


EVENT_ORDER = [
    "hypoxemia",
    "ventilation_etco2",
    "hypotension",
    "hypertension",
    "depth_of_anesthesia",
    "bradycardia",
    "tachycardia",
    "hemorrhage_or_volume",
    "general_warning",
]


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def clinical_assessment(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    obj = snapshot.get("clinical_assessment", {})
    return obj if isinstance(obj, dict) else {}


def recent_state(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    obj = clinical_assessment(snapshot).get("recent_state_mean", {})
    return obj if isinstance(obj, dict) else {}


def persistence(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    obj = clinical_assessment(snapshot).get("persistence_seconds", {})
    return obj if isinstance(obj, dict) else {}


def alarm_tags(snapshot: Dict[str, Any]) -> List[str]:
    tags = clinical_assessment(snapshot).get("alarm_tags", [])
    return [str(x) for x in tags] if isinstance(tags, list) else []


def extract_snapshot(row_or_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(row_or_snapshot.get("snapshot"), dict):
        return row_or_snapshot["snapshot"]
    return row_or_snapshot


def extract_features(row_or_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = extract_snapshot(row_or_snapshot)
    recent = recent_state(snapshot)
    persist = persistence(snapshot)
    clinical = clinical_assessment(snapshot)
    features: Dict[str, Any] = {}

    numeric_keys = {
        "HR_bpm": recent.get("HR_bpm"),
        "MAP_mmhg": recent.get("MAP_mmhg"),
        "SBP_mmhg": recent.get("SBP_mmhg"),
        "DBP_mmhg": recent.get("DBP_mmhg"),
        "SpO2_pct": recent.get("SpO2_pct"),
        "EtCO2_mmhg": recent.get("EtCO2_mmhg"),
        "BIS": recent.get("BIS"),
        "SVV_pct": recent.get("SVV_pct"),
        "CVP_mmhg": recent.get("CVP_mmhg"),
        "CO_L_min": recent.get("CO_L_min"),
        "CI_L_min_m2": recent.get("CI_L_min_m2"),
        "etco2_missing": persist.get("etco2_missing"),
        "etco2_zero_like": persist.get("etco2_zero_like"),
        "spo2_lt_90": persist.get("spo2_lt_90"),
        "map_lt_65": persist.get("map_lt_65"),
        "map_lt_55": persist.get("map_lt_55"),
        "hr_lt_50": persist.get("hr_lt_50"),
        "hr_gt_100": persist.get("hr_gt_100"),
        "bis_gt_60": persist.get("bis_gt_60"),
        "bis_lt_40": persist.get("bis_lt_40"),
        "svv_ge_13": persist.get("svv_ge_13"),
        "cvp_le_2": persist.get("cvp_le_2"),
        "cvp_ge_15": persist.get("cvp_ge_15"),
    }
    for key, value in numeric_keys.items():
        num = _as_float(value)
        features[key] = -999.0 if num is None else num
        features[f"has_{key}"] = 0 if num is None else 1

    for tag in alarm_tags(snapshot):
        features[f"alarm={tag}"] = 1

    for event in clinical.get("adverse_event_types", []) or []:
        features[f"adverse={event}"] = 1

    features[f"sample_category={snapshot.get('sample_category', '')}"] = 1
    features[f"risk_level={clinical.get('risk_level', '')}"] = 1
    features[f"surgery_group={snapshot.get('surgery_group', '')}"] = 1
    features[f"surgery_type={snapshot.get('surgery_type', '')}"] = 1
    features[f"interpreted_intervention={snapshot.get('interpreted_intervention_type', '')}"] = 1
    return features


def derive_event_label(row_or_snapshot: Dict[str, Any]) -> str:
    snapshot = extract_snapshot(row_or_snapshot)
    clinical = clinical_assessment(snapshot)
    for key in ("label_primary_event", "primary_event", "event_type"):
        value = row_or_snapshot.get(key) or snapshot.get(key) or clinical.get(key)
        if isinstance(value, str) and value:
            return value

    recent = recent_state(snapshot)
    alarms = set(alarm_tags(snapshot))
    spo2 = _as_float(recent.get("SpO2_pct"))
    map_v = _as_float(recent.get("MAP_mmhg"))
    sbp = _as_float(recent.get("SBP_mmhg"))
    dbp = _as_float(recent.get("DBP_mmhg"))
    hr = _as_float(recent.get("HR_bpm"))
    etco2 = _as_float(recent.get("EtCO2_mmhg"))
    bis = _as_float(recent.get("BIS"))
    svv = _as_float(recent.get("SVV_pct"))
    persist = persistence(snapshot)
    etco2_missing = _as_float(persist.get("etco2_missing")) or 0.0
    etco2_zero = _as_float(persist.get("etco2_zero_like")) or 0.0

    if spo2 is not None and spo2 < 94 or "SpO2" in alarms:
        return "hypoxemia"
    if "EtCO2" in alarms or etco2_missing > 0 or etco2_zero > 0 or (etco2 is not None and (etco2 < 25 or etco2 > 55)):
        return "ventilation_etco2"
    if map_v is not None and map_v < 65 or "MAP" in alarms:
        return "hypotension"
    if (sbp is not None and sbp > 180) or (dbp is not None and dbp > 100) or "SBP" in alarms or "DBP" in alarms:
        return "hypertension"
    if bis is not None and bis > 60 or "BIS" in alarms:
        return "depth_of_anesthesia"
    if hr is not None and hr < 50:
        return "bradycardia"
    if hr is not None and hr > 100:
        return "tachycardia"
    if svv is not None and svv >= 13:
        return "hemorrhage_or_volume"
    return "general_warning"


def derive_severity_label(row_or_snapshot: Dict[str, Any]) -> str:
    snapshot = extract_snapshot(row_or_snapshot)
    clinical = clinical_assessment(snapshot)
    for key in ("label_severity", "severity"):
        value = row_or_snapshot.get(key) or snapshot.get(key) or clinical.get(key)
        if isinstance(value, str) and value:
            return value

    category = str(snapshot.get("sample_category", "")).lower()
    risk = str(clinical.get("risk_level", "")).lower()
    if category in {"critical_alarm", "critical"} or risk in {"critical", "high"}:
        return "critical"
    if category in {"warning_signal", "warning"} or risk in {"moderate", "warning"}:
        return "warning"
    return "stable"


def load_router(path: str) -> Dict[str, Any]:
    if joblib is None:
        raise RuntimeError("joblib is required to load trained router models")
    return joblib.load(path)


def predict_router(model_bundle: Dict[str, Any], row_or_snapshot: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    features = extract_features(row_or_snapshot)
    vectorizer = model_bundle["vectorizer"]
    x = vectorizer.transform([features])
    event = str(model_bundle["event_model"].predict(x)[0])
    severity = str(model_bundle["severity_model"].predict(x)[0])
    details: Dict[str, Any] = {"features_used": len(features)}
    if hasattr(model_bundle["event_model"], "predict_proba"):
        classes = list(model_bundle["event_model"].classes_)
        probs = model_bundle["event_model"].predict_proba(x)[0]
        details["event_probabilities"] = {str(c): float(p) for c, p in zip(classes, probs)}
    if hasattr(model_bundle["severity_model"], "predict_proba"):
        classes = list(model_bundle["severity_model"].classes_)
        probs = model_bundle["severity_model"].predict_proba(x)[0]
        details["severity_probabilities"] = {str(c): float(p) for c, p in zip(classes, probs)}
    return event, severity, details
