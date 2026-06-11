"""Multimodal operating-room anesthesia agent workflow.

This workflow maps the agent system to the real operating-room collaboration
pattern:

- Perception Agent: "eyes", like the circulating nurse / low-latency monitor.
- Decision Agent: "brain", like the attending anesthesiologist.
- Evaluation & Reflection Agent: quality-control reviewer / safety judge.

The current implementation is dataset-ready: it consumes the thoracic anesthesia
snapshot JSON generated from monitor trends, clinical context, Miller retrieval,
and optional waveform image paths. The fast/slow perception nodes are designed
as replaceable adapters for OCR, lightweight visual encoders, or VLMs.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

from .trained_event_router import load_router, predict_router
from .vision_interfaces import monitor_snapshot_ocr, run_slow_vision_adapter
from .crisis_workflow import (
    CRISIS_EVENT_ORDER,
    EVENT_AGENT_NAMES,
    normalize_dataset_row,
)


class ORAgentState(TypedDict, total=False):
    case: Dict[str, Any]
    snapshot: Dict[str, Any]
    config: Dict[str, Any]
    blackboard: Dict[str, Any]
    perception: Dict[str, Any]
    decision_mode: str
    decision: Dict[str, Any]
    evaluation: Dict[str, Any]
    final_report: Dict[str, Any]
    audit_trail: List[Dict[str, Any]]
    high_priority_alert: bool
    human_review_required: bool


@dataclass
class FastStreamOutput:
    source: str = "dataset_snapshot_fast_stream"
    extracted_vitals: Dict[str, Optional[float]] = field(default_factory=dict)
    alarm_tags: List[str] = field(default_factory=list)
    extreme_waveform_flags: List[str] = field(default_factory=list)
    high_priority_alerts: List[str] = field(default_factory=list)
    latency_class: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SlowStreamOutput:
    source: str = "dataset_snapshot_slow_stream"
    patient_context: Dict[str, Any] = field(default_factory=dict)
    structured_context: Dict[str, Any] = field(default_factory=dict)
    image_descriptions: List[str] = field(default_factory=list)
    knowledge_terms: List[str] = field(default_factory=list)
    evidence_context: List[str] = field(default_factory=list)
    latency_class: str = "high"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _clinical_assessment(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    obj = snapshot.get("clinical_assessment", {})
    return obj if isinstance(obj, dict) else {}


def _recent(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    obj = _clinical_assessment(snapshot).get("recent_state_mean", {})
    return obj if isinstance(obj, dict) else {}


def _persistence(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    obj = _clinical_assessment(snapshot).get("persistence_seconds", {})
    return obj if isinstance(obj, dict) else {}


def _alarm_tags(snapshot: Dict[str, Any]) -> List[str]:
    tags = _clinical_assessment(snapshot).get("alarm_tags", [])
    return [str(x) for x in tags] if isinstance(tags, list) else []


def _risk_flags(snapshot: Dict[str, Any]) -> List[str]:
    flags = _clinical_assessment(snapshot).get("risk_flags", [])
    return [str(x) for x in flags] if isinstance(flags, list) else []


def _severity_rank(severity: str) -> int:
    return {"critical": 4, "rescue": 4, "high": 3, "warning": 2, "moderate": 2, "stable": 1, "low": 1}.get(
        severity.lower(), 1
    )


class MultimodalORAnesthesiaWorkflow:
    """Three-agent OR workflow using LangGraph StateGraph when available."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self.trained_router = None
        router_path = str(self.config.get("trained_event_router_path", "") or "")
        if router_path:
            try:
                self.trained_router = load_router(router_path)
            except Exception as exc:
                if bool(self.config.get("require_trained_router", False)):
                    raise
                self.config["trained_event_router_error"] = str(exc)

    def initial_state(self, row_or_snapshot: Dict[str, Any]) -> ORAgentState:
        normalized = normalize_dataset_row(row_or_snapshot)
        return {
            "case": normalized["case"],
            "snapshot": normalized["snapshot"],
            "config": dict(self.config),
            "blackboard": {
                "observations": [],
                "hypotheses": [],
                "recommendations": [],
                "conflicts": [],
                "safety_constraints": [],
            },
            "perception": {},
            "decision_mode": "routine",
            "decision": {},
            "evaluation": {},
            "final_report": {},
            "audit_trail": [],
            "high_priority_alert": False,
            "human_review_required": False,
        }

    def invoke(self, row_or_snapshot: Dict[str, Any]) -> ORAgentState:
        return self.compile().invoke(self.initial_state(row_or_snapshot))

    def compile(self) -> Any:
        try:
            from langgraph.graph import END, START, StateGraph  # type: ignore

            graph = StateGraph(ORAgentState)
            graph.add_node("fast_perception_stream", self.fast_perception_stream)
            graph.add_node("slow_perception_stream", self.slow_perception_stream)
            graph.add_node("perception_fusion", self.perception_fusion)
            graph.add_node("decision_router", self.decision_router)
            graph.add_node("emergency_decision", self.emergency_decision)
            graph.add_node("routine_decision", self.routine_decision)
            graph.add_node("evaluation_reflection", self.evaluation_reflection)
            graph.add_node("final_safety_gate", self.final_safety_gate)

            graph.add_edge(START, "fast_perception_stream")
            graph.add_edge("fast_perception_stream", "slow_perception_stream")
            graph.add_edge("slow_perception_stream", "perception_fusion")
            graph.add_edge("perception_fusion", "decision_router")
            graph.add_conditional_edges(
                "decision_router",
                self._route_decision_mode,
                {"emergency": "emergency_decision", "routine": "routine_decision"},
            )
            graph.add_edge("emergency_decision", "evaluation_reflection")
            graph.add_edge("routine_decision", "evaluation_reflection")
            graph.add_edge("evaluation_reflection", "final_safety_gate")
            graph.add_edge("final_safety_gate", END)
            return graph.compile()
        except Exception:
            return _SequentialORApp(self)

    # 1. Perception Agent, Fast Stream
    def fast_perception_stream(self, state: ORAgentState) -> ORAgentState:
        next_state = deepcopy(state)
        snapshot = next_state.get("snapshot", {})
        persist = _persistence(snapshot)
        fast_adapter = monitor_snapshot_ocr(snapshot)
        vitals = fast_adapter.get("extracted_vitals", {})
        alarms = fast_adapter.get("alarm_tags", _alarm_tags(snapshot))
        waveform_flags: List[str] = []
        high_alerts: List[str] = []

        if vitals["SpO2_pct"] is not None and vitals["SpO2_pct"] < 90:
            high_alerts.append("critical_hypoxemia")
        if vitals["MAP_mmhg"] is not None and vitals["MAP_mmhg"] < 55:
            high_alerts.append("critical_hypotension")
        if vitals["HR_bpm"] is not None and vitals["HR_bpm"] < 40:
            high_alerts.append("severe_bradycardia")
        if vitals["HR_bpm"] is not None and vitals["HR_bpm"] > 140:
            high_alerts.append("severe_tachycardia")

        etco2_missing = _as_float(persist.get("etco2_missing")) or 0.0
        etco2_zero = _as_float(persist.get("etco2_zero_like")) or 0.0
        if "EtCO2" in alarms or etco2_missing > 0 or etco2_zero > 0:
            waveform_flags.append("etco2_signal_loss_or_sampling_issue")
            if etco2_zero >= 10 or etco2_missing >= 10:
                high_alerts.append("persistent_etco2_loss")

        category = str(snapshot.get("sample_category", "")).lower()
        risk_level = str(_clinical_assessment(snapshot).get("risk_level", "")).lower()
        if category in {"critical_alarm", "critical"} or risk_level in {"high", "critical"}:
            high_alerts.append("dataset_critical_alarm")

        fast = FastStreamOutput(
            extracted_vitals=vitals,
            alarm_tags=alarms,
            extreme_waveform_flags=_dedupe(waveform_flags),
            high_priority_alerts=_dedupe(high_alerts),
        )
        perception = dict(next_state.get("perception", {}))
        fast_dict = fast.to_dict()
        fast_dict["vision_adapter"] = fast_adapter
        perception["fast_stream"] = fast_dict
        next_state["perception"] = perception
        next_state["high_priority_alert"] = bool(high_alerts)
        next_state["audit_trail"] = list(next_state.get("audit_trail", [])) + [
            {
                "stage": "fast_perception_stream",
                "role_mapping": "Perception Agent / circulating nurse",
                "high_priority_alerts": fast.high_priority_alerts,
                "alarm_tags": alarms,
            }
        ]
        return next_state

    # 1. Perception Agent, Slow Stream
    def slow_perception_stream(self, state: ORAgentState) -> ORAgentState:
        next_state = deepcopy(state)
        snapshot = next_state.get("snapshot", {})
        clinical = _clinical_assessment(snapshot)
        patient_context = {
            "patient_background": snapshot.get("patient_background", {}),
            "preop_context": snapshot.get("preop_context", []),
            "surgery_type": snapshot.get("surgery_type", ""),
            "intraop_stage": snapshot.get("intraop_stage", ""),
            "surgery_group": snapshot.get("surgery_group", next_state.get("case", {}).get("surgery_group")),
        }
        image_descriptions = []
        if snapshot.get("waveform_image_path"):
            image_descriptions.append(f"waveform_image_available: {snapshot.get('waveform_image_path')}")
        knowledge_terms = _dedupe(
            [str(x) for x in clinical.get("alarm_tags", [])]
            + [str(x) for x in clinical.get("adverse_event_types", [])]
            + [str(snapshot.get("interpreted_intervention_type", ""))]
        )
        evidence_context = []
        alignment = snapshot.get("miller_alignment", {})
        if isinstance(alignment, dict):
            if alignment.get("verdict"):
                evidence_context.append(f"Miller/VitalDB verdict: {alignment.get('verdict')}")
            if alignment.get("reason"):
                evidence_context.append(f"Alignment reason: {alignment.get('reason')}")
        if snapshot.get("miller_retrieval") or next_state.get("case", {}).get("miller_retrieval"):
            evidence_context.append("Miller retrieval context available")

        vlm_result = run_slow_vision_adapter(snapshot, self.config)
        slow = SlowStreamOutput(
            patient_context=patient_context,
            structured_context={
                "vital_trend_last_5min": snapshot.get("vital_trend_last_5min", {}),
                "baseline_comparison": snapshot.get("baseline_comparison", {}),
                "risk_flags": _risk_flags(snapshot),
                "actual_intervention_bundle": snapshot.get("actual_intervention_bundle", ""),
                "concurrent_medications_active": snapshot.get("concurrent_medications_active", []),
            },
            image_descriptions=image_descriptions,
            knowledge_terms=knowledge_terms,
            evidence_context=evidence_context,
        )
        perception = dict(next_state.get("perception", {}))
        slow_dict = slow.to_dict()
        slow_dict["vlm_adapter"] = vlm_result
        perception["slow_stream"] = slow_dict
        next_state["perception"] = perception
        next_state["audit_trail"] = list(next_state.get("audit_trail", [])) + [
            {
                "stage": "slow_perception_stream",
                "role_mapping": "Perception Agent / deep multimodal channel",
                "knowledge_terms": knowledge_terms,
                "evidence_context": evidence_context,
            }
        ]
        return next_state

    def perception_fusion(self, state: ORAgentState) -> ORAgentState:
        next_state = deepcopy(state)
        perception = dict(next_state.get("perception", {}))
        fast = perception.get("fast_stream", {})
        slow = perception.get("slow_stream", {})
        events = self._detect_events_from_perception(fast, slow, next_state.get("snapshot", {}))
        primary = self._select_primary_event(events)
        trained_router_info = {}
        if self.trained_router is not None:
            model_event, model_severity, model_details = predict_router(
                self.trained_router,
                {"case": next_state.get("case", {}), "snapshot": next_state.get("snapshot", {})},
            )
            primary = {
                "event_type": model_event,
                "severity": model_severity,
                "evidence": "trained_event_router_prediction",
            }
            trained_router_info = {
                "enabled": True,
                "predicted_event": model_event,
                "predicted_severity": model_severity,
                "details": model_details,
            }
        perception["trained_router"] = trained_router_info or {"enabled": False}
        perception["fused_context"] = {
            "events": events,
            "primary_event": primary.get("event_type", "general_warning"),
            "severity": primary.get("severity", "stable"),
            "route_reason": primary.get("evidence", ""),
            "structured_multimodal_context": {
                "vitals": fast.get("extracted_vitals", {}),
                "alarm_tags": fast.get("alarm_tags", []),
                "waveform_flags": fast.get("extreme_waveform_flags", []),
                "patient_context": slow.get("patient_context", {}),
                "risk_flags": slow.get("structured_context", {}).get("risk_flags", []),
                "image_descriptions": slow.get("image_descriptions", []),
            },
        }
        next_state["perception"] = perception
        blackboard = dict(next_state.get("blackboard", {}))
        blackboard["observations"] = _dedupe(
            list(blackboard.get("observations", []))
            + [
                f"primary_event={primary.get('event_type', 'general_warning')}",
                f"severity={primary.get('severity', 'stable')}",
                f"route_reason={primary.get('evidence', '')}",
            ]
        )
        next_state["blackboard"] = blackboard
        next_state["high_priority_alert"] = bool(
            next_state.get("high_priority_alert", False) or _severity_rank(str(primary.get("severity", ""))) >= 3
        )
        next_state["audit_trail"] = list(next_state.get("audit_trail", [])) + [
            {
                "stage": "perception_fusion",
                "primary_event": perception["fused_context"]["primary_event"],
                "severity": perception["fused_context"]["severity"],
            }
        ]
        return next_state

    # 2. Decision Agent
    def decision_router(self, state: ORAgentState) -> ORAgentState:
        next_state = deepcopy(state)
        fused = next_state.get("perception", {}).get("fused_context", {})
        severity = str(fused.get("severity", "stable"))
        high_priority = bool(next_state.get("high_priority_alert", False))
        mode = "emergency" if high_priority or _severity_rank(severity) >= 3 else "routine"
        next_state["decision_mode"] = mode
        next_state["audit_trail"] = list(next_state.get("audit_trail", [])) + [
            {
                "stage": "decision_router",
                "agent": "Decision Agent",
                "mode": mode,
                "reason": fused.get("route_reason", ""),
                "policy": "critical/high severity enters emergency mode; stable/warning enters routine monitoring mode",
            }
        ]
        return next_state
    def emergency_decision(self, state: ORAgentState) -> ORAgentState:
        next_state = deepcopy(state)
        fused = next_state.get("perception", {}).get("fused_context", {})
        primary_event = str(fused.get("primary_event", "general_warning"))
        decision = self._make_decision(primary_event, "emergency", fused)
        decision["internal_clinical_modules"] = self._decision_internal_modules(primary_event, fused)
        next_state["decision"] = decision
        next_state["human_review_required"] = True
        next_state["audit_trail"] = list(next_state.get("audit_trail", [])) + [
            {
                "stage": "emergency_decision",
                "agent": "Decision Agent",
                "mode": "emergency",
                "primary_event": primary_event,
                "intervention_count": len(decision.get("intervention_plan", [])),
            }
        ]
        return next_state
    def routine_decision(self, state: ORAgentState) -> ORAgentState:
        next_state = deepcopy(state)
        fused = next_state.get("perception", {}).get("fused_context", {})
        primary_event = str(fused.get("primary_event", "general_warning"))
        decision = self._make_decision(primary_event, "routine", fused)
        decision["internal_clinical_modules"] = self._decision_internal_modules(primary_event, fused)
        next_state["decision"] = decision
        next_state["audit_trail"] = list(next_state.get("audit_trail", [])) + [
            {
                "stage": "routine_decision",
                "agent": "Decision Agent",
                "mode": "routine",
                "primary_event": primary_event,
                "intervention_count": len(decision.get("intervention_plan", [])),
            }
        ]
        return next_state

    # 3. Evaluation & Reflection Agent
    def evaluation_reflection(self, state: ORAgentState) -> ORAgentState:
        next_state = deepcopy(state)
        decision = next_state.get("decision", {})
        perception = next_state.get("perception", {})
        fused = perception.get("fused_context", {})
        snapshot = next_state.get("snapshot", {})
        alignment = snapshot.get("miller_alignment", {})
        hard_failures: List[str] = []
        scores = {
            "diagnosis_accuracy": 3,
            "priority_order": 3,
            "intervention_safety": 3,
            "dose_or_action_specificity": 2,
            "logic_consistency": 3,
        }

        primary_event = str(fused.get("primary_event", "general_warning"))
        plan_text = json.dumps(decision.get("intervention_plan", []), ensure_ascii=False)
        if primary_event in {"hypoxemia", "ventilation_etco2"} and "气道" in plan_text:
            scores["priority_order"] = 5
            scores["intervention_safety"] = 4
        if primary_event == "hypotension" and "加深麻醉" in plan_text:
            hard_failures.append("低血压路径中不应优先加深麻醉")
        if primary_event == "depth_of_anesthesia" and any(x in plan_text for x in ("低氧", "低灌注", "通气")):
            scores["logic_consistency"] = 4
        if isinstance(alignment, dict) and alignment.get("high_risk_conflict"):
            hard_failures.append("Miller/VitalDB alignment 标记 high_risk_conflict")
        if str(fused.get("severity", "")).lower() in {"critical", "high"}:
            scores["priority_order"] = max(scores["priority_order"], 4)

        reflection = []
        if hard_failures:
            reflection.append("触发安全红线，必须人工复核并修正方案。")
        if not decision.get("reassessment_plan"):
            reflection.append("缺少复评时间窗，应补充 30-60 秒或 3-5 分钟复评节点。")
        if not reflection:
            reflection.append("方案包含事件识别、优先级、干预和复评，可作为人工复核草案。")

        evaluation = {
            "agent": "Evaluation & Reflection Agent",
            "role_mapping": "medical quality-control judge",
            "hard_failures": hard_failures,
            "scores_0_to_5": scores,
            "overall_score": round(sum(scores.values()) / max(1, len(scores)), 2),
            "reflection": reflection,
            "gold_reference_available": bool(alignment),
            "alignment_verdict": alignment.get("verdict") if isinstance(alignment, dict) else "",
        }
        next_state["evaluation"] = evaluation
        if hard_failures:
            next_state["human_review_required"] = True
        next_state["audit_trail"] = list(next_state.get("audit_trail", [])) + [
            {
                "stage": "evaluation_reflection",
                "hard_failures": hard_failures,
                "overall_score": evaluation["overall_score"],
            }
        ]
        return next_state

    def final_safety_gate(self, state: ORAgentState) -> ORAgentState:
        next_state = deepcopy(state)
        fused = next_state.get("perception", {}).get("fused_context", {})
        evaluation = next_state.get("evaluation", {})
        safety_red_lines = list(evaluation.get("hard_failures", []))
        if str(fused.get("severity", "")).lower() in {"critical", "high"}:
            safety_red_lines.append("危急场景必须由麻醉医生立即确认，Agent 不可自动执行医嘱。")
        if next_state.get("high_priority_alert", False):
            safety_red_lines.append("Fast Stream 触发零延迟高优先级报警。")
        primary_event = str(fused.get("primary_event", "general_warning"))
        severity = str(fused.get("severity", "stable")).lower()
        event_requires_review = primary_event != "general_warning" or severity not in {"stable", "low", ""}
        human_review = bool(next_state.get("human_review_required", False) or safety_red_lines or event_requires_review)
        next_state["human_review_required"] = human_review
        report = {
            "caseid": next_state.get("case", {}).get("caseid"),
            "workflow": "multimodal_or_anesthesia_agent",
            "architecture": {
                "agent_system": "three_agent_multimodal_anesthesia_system",
                "perception_agent": "Fast Stream + Slow Stream + trained event router for multimodal clinical context structuring",
                "decision_agent": "emergency rescue mode + routine monitoring mode + internal clinical modules",
                "evaluation_agent": "quality scoring + reflection + final safety gate",
                "orchestration": "LangGraph StateGraph with deterministic fallback; only three exposed agents: Perception, Decision, Evaluation",
            },            "blackboard": next_state.get("blackboard", {}),
            "perception": next_state.get("perception", {}),
            "decision_mode": next_state.get("decision_mode", ""),
            "decision": next_state.get("decision", {}),
            "evaluation": evaluation,
            "safety_red_lines": _dedupe(safety_red_lines),
            "human_review_required": human_review,
            "clinical_safety_notice": (
                "本系统用于围术期麻醉多模态决策支持和研究评估，不得作为自动诊疗、自动医嘱或自动给药依据。"
                "所有输出必须由具备资质的麻醉医生结合实时监护、设备状态、本院规范和药品说明书确认。"
            ),
        }
        next_state["final_report"] = report
        next_state["audit_trail"] = list(next_state.get("audit_trail", [])) + [
            {"stage": "final_safety_gate", "human_review_required": human_review}
        ]
        return next_state

    def _route_decision_mode(self, state: ORAgentState) -> str:
        mode = str(state.get("decision_mode", "routine"))
        return "emergency" if mode == "emergency" else "routine"


    def _decision_internal_modules(self, primary_event: str, fused: Dict[str, Any]) -> Dict[str, Any]:
        """Clinical submodules inside the Decision Agent, not exposed as separate agents."""
        context = fused.get("structured_multimodal_context", {})
        vitals = context.get("vitals", {})
        severity = str(fused.get("severity", "")).lower()
        return {
            "airway_ventilation_check": {
                "enabled": primary_event in {"hypoxemia", "ventilation_etco2"},
                "focus": [
                    "确认气管导管/双腔管/支气管封堵器位置。",
                    "检查 EtCO2 采样管、过滤器、呼吸回路漏气或堵塞。",
                    "确认胸廓起伏、气道压、潮气量和分钟通气量。",
                ],
            },
            "hemodynamic_check": {
                "enabled": primary_event in {"hypotension", "hypertension", "bradycardia", "tachycardia"},
                "focus": [
                    f"结合 MAP={vitals.get('MAP_mmhg')}、HR={vitals.get('HR_bpm')} 判断灌注状态。",
                    "确认血压测量可靠性，结合容量、出血、麻醉深度和手术刺激判断原因。",
                ],
            },
            "depth_and_drug_safety_check": {
                "enabled": True,
                "focus": [
                    f"BIS={vitals.get('BIS')} 仅作为辅助信号，不能单独触发大幅给药。",
                    "所有药物调整必须由麻醉医生确认剂量、单位、禁忌证和当前循环/通气状态。",
                ],
            },
            "crisis_protocol_check": {
                "enabled": severity in {"critical", "high"},
                "focus": [
                    "危急事件按 ABC、氧合、通气和灌注优先级处理。",
                    "设置 30-60 秒快速复评和 3-5 分钟趋势复评。",
                ],
            },
        }

    def _detect_events_from_perception(
        self,
        fast: Dict[str, Any],
        slow: Dict[str, Any],
        snapshot: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        vitals = fast.get("extracted_vitals", {})
        alarms = set(fast.get("alarm_tags", []))
        flags = set(fast.get("extreme_waveform_flags", []))
        high_alerts = set(fast.get("high_priority_alerts", []))
        events: List[Dict[str, Any]] = []

        spo2 = _as_float(vitals.get("SpO2_pct"))
        map_v = _as_float(vitals.get("MAP_mmhg"))
        sbp = _as_float(vitals.get("SBP_mmhg"))
        dbp = _as_float(vitals.get("DBP_mmhg"))
        hr = _as_float(vitals.get("HR_bpm"))
        etco2 = _as_float(vitals.get("EtCO2_mmhg"))
        bis = _as_float(vitals.get("BIS"))
        svv = _as_float(vitals.get("SVV_pct"))

        contextual_critical = "dataset_critical_alarm" in high_alerts
        if spo2 is not None and spo2 < 90:
            events.append({"event_type": "hypoxemia", "severity": "critical", "evidence": f"SpO2={spo2}"})
        elif spo2 is not None and (spo2 < 94 or "SpO2" in alarms):
            events.append({"event_type": "hypoxemia", "severity": "warning", "evidence": f"SpO2={spo2} or alarm"})

        if "EtCO2" in alarms or "etco2_signal_loss_or_sampling_issue" in flags or (etco2 is not None and etco2 < 25):
            events.append(
                {
                    "event_type": "ventilation_etco2",
                    "severity": "critical" if contextual_critical or "persistent_etco2_loss" in high_alerts else "warning",
                    "evidence": f"EtCO2={etco2}, flags={list(flags)}",
                }
            )
        if map_v is not None and map_v < 55:
            events.append({"event_type": "hypotension", "severity": "critical", "evidence": f"MAP={map_v}"})
        elif map_v is not None and (map_v < 65 or "MAP" in alarms):
            events.append({"event_type": "hypotension", "severity": "warning", "evidence": f"MAP={map_v} or alarm"})
        if (sbp is not None and sbp > 180) or (dbp is not None and dbp > 100) or "SBP" in alarms or "DBP" in alarms:
            events.append(
                {
                    "event_type": "hypertension",
                    "severity": "critical" if contextual_critical else "warning",
                    "evidence": f"SBP={sbp}, DBP={dbp}, alarm={bool({'SBP','DBP'} & alarms)}",
                }
            )
        if bis is not None and bis > 60 or "BIS" in alarms:
            events.append(
                {
                    "event_type": "depth_of_anesthesia",
                    "severity": "critical" if contextual_critical else "warning",
                    "evidence": f"BIS={bis}, alarm={'BIS' in alarms}",
                }
            )
        if hr is not None and hr < 50:
            events.append(
                {
                    "event_type": "bradycardia",
                    "severity": "critical" if hr < 45 else "warning",
                    "evidence": f"HR={hr}",
                }
            )
        if hr is not None and hr > 100:
            events.append(
                {
                    "event_type": "tachycardia",
                    "severity": "critical" if hr > 120 else "warning",
                    "evidence": f"HR={hr}",
                }
            )
        if svv is not None and svv >= 13:
            events.append({"event_type": "hemorrhage_or_volume", "severity": "warning", "evidence": f"SVV={svv}"})
        if not events:
            category = str(snapshot.get("sample_category", "stable"))
            risk = str(_clinical_assessment(snapshot).get("risk_level", "stable"))
            events.append({"event_type": "general_warning", "severity": risk, "evidence": f"sample_category={category}"})
        return events

    def _select_primary_event(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        priority = {event: i for i, event in enumerate(CRISIS_EVENT_ORDER)}
        return sorted(
            events,
            key=lambda e: (
                -_severity_rank(str(e.get("severity", "stable"))),
                priority.get(str(e.get("event_type", "general_warning")), 999),
            ),
        )[0]

    def _make_decision(self, event_type: str, mode: str, fused: Dict[str, Any]) -> Dict[str, Any]:
        context = fused.get("structured_multimodal_context", {})
        base = {
            "agent": "Decision Agent",
            "role_mapping": "attending anesthesiologist",
            "mode": mode,
            "primary_event": event_type,
            "reasoning_style": "truncated rescue reasoning" if mode == "emergency" else "long-horizon causal reasoning",
            "diagnostic_hypothesis": "",
            "intervention_plan": [],
            "self_check": [],
            "reassessment_plan": [],
            "context_used": context,
        }
        if event_type == "ventilation_etco2":
            base.update(
                {
                    "diagnostic_hypothesis": "EtCO2 信号丢失/采样管或呼吸回路异常，需先排除真实通气中断。",
                    "intervention_plan": [
                        "立即检查 EtCO2 采样管、过滤器、气管导管连接和呼吸回路完整性。",
                        "确认胸廓起伏、气道压、呼吸机参数和 SpO2 趋势。",
                        "胸科/单肺通气场景下复核双腔管或支气管封堵器位置。",
                        "通气安全确认后，再评估是否需要调整麻醉深度或镇痛。",
                    ],
                    "self_check": ["避免在通气未确认前单纯加深麻醉。", "若合并低氧或循环崩溃，立即升级抢救。"],
                    "reassessment_plan": ["30-60 秒内复核 EtCO2 波形", "1-3 分钟内复核 SpO2/HR/MAP 变化"],
                }
            )
        elif event_type == "hypoxemia":
            base.update(
                {
                    "diagnostic_hypothesis": "围术期低氧，需优先处理氧合、气道和单肺通气相关可逆原因。",
                    "intervention_plan": [
                        "提高氧合支持并确认 FiO2、气道压力和潮气量。",
                        "检查导管位置、肺隔离状态、分泌物/痰栓和手术压迫因素。",
                        "必要时请求暂停手术刺激并组织麻醉-外科协同处理。",
                    ],
                    "self_check": ["低氧处理优先于镇静/降压等非救命目标。"],
                    "reassessment_plan": ["1 分钟内复核 SpO2/EtCO2", "必要时复查 ABG"],
                }
            )
        elif event_type == "hypotension":
            base.update(
                {
                    "diagnostic_hypothesis": "低血压/低灌注，需区分容量、麻醉深度、出血、过敏和心功能因素。",
                    "intervention_plan": [
                        "确认血压信号可靠性和换能器/袖带状态。",
                        "快速评估出血、容量反应性、HR、麻醉深度和用药变化。",
                        "由麻醉医生确认补液、血管活性药或暂停加深麻醉策略。",
                    ],
                    "self_check": ["MAP 低时不应优先增加丙泊酚或阿片。"],
                    "reassessment_plan": ["30-60 秒复核 MAP/HR", "3 分钟内评估灌注改善"],
                }
            )
        elif event_type == "hypertension":
            base.update(
                {
                    "diagnostic_hypothesis": "高血压/交感反应，可能与刺激、镇痛不足、麻醉浅或通气异常相关。",
                    "intervention_plan": [
                        "确认血压波形可靠性。",
                        "结合 BIS、HR、EtCO2 和手术刺激判断诱因。",
                        "先处理低氧/通气异常，再考虑镇痛、镇静或降压策略。",
                    ],
                    "self_check": ["避免在低灌注边界附近过度降压。"],
                    "reassessment_plan": ["1-3 分钟复核 BP/HR/BIS/EtCO2"],
                }
            )
        elif event_type == "depth_of_anesthesia":
            base.update(
                {
                    "diagnostic_hypothesis": "麻醉深度不足或 BIS 信号异常，需要结合临床体征和泵状态综合判断。",
                    "intervention_plan": [
                        "确认 BIS 电极质量、肌电干扰和泵路运行。",
                        "若通气和循环稳定，由麻醉医生小步调整镇静/镇痛。",
                        "若合并低血压/低氧，优先处理灌注或氧合。",
                    ],
                    "self_check": ["BIS 仅作为支持信号，不能单独决定给药。"],
                    "reassessment_plan": ["1 分钟复核 BIS 与生命体征", "3-5 分钟复核趋势"],
                }
            )
        else:
            base.update(
                {
                    "diagnostic_hypothesis": f"当前主事件为 {EVENT_AGENT_NAMES.get(event_type, event_type)}，需要结合趋势继续观察。",
                    "intervention_plan": ["复核监测信号质量。", "结合手术阶段和既往干预历史小步调整。", "保留人工复核。"],
                    "self_check": ["不自动执行医嘱。"],
                    "reassessment_plan": ["3-5 分钟后复核趋势"],
                }
            )
        if mode == "emergency":
            base["intervention_plan"] = ["触发抢救模式：截断冗长推理，优先处理 ABC/灌注安全。"] + base[
                "intervention_plan"
            ]
        return base


class _SequentialORApp:
    def __init__(self, workflow: MultimodalORAnesthesiaWorkflow) -> None:
        self.workflow = workflow

    def invoke(self, state: ORAgentState) -> ORAgentState:
        state = self.workflow.fast_perception_stream(state)
        state = self.workflow.slow_perception_stream(state)
        state = self.workflow.perception_fusion(state)
        state = self.workflow.decision_router(state)
        if self.workflow._route_decision_mode(state) == "emergency":
            state = self.workflow.emergency_decision(state)
        else:
            state = self.workflow.routine_decision(state)
        state = self.workflow.evaluation_reflection(state)
        state = self.workflow.final_safety_gate(state)
        return state


def run_multimodal_or_workflow(
    row_or_snapshot: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    workflow = MultimodalORAnesthesiaWorkflow(config=config)
    state = workflow.invoke(row_or_snapshot)
    report = state.get("final_report", {})
    report["audit_trail"] = state.get("audit_trail", [])
    return report












