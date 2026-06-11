"""Event-driven perioperative anesthesia crisis workflow.

This module is the workflow that matches the thoracic anesthesia dataset:
case context + intraoperative vital snapshots + alert tags + intervention
alignment. LangGraph is used for orchestration, while model training can later
be attached to the detector/router nodes.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypedDict


CRISIS_EVENT_ORDER: List[str] = [
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


EVENT_AGENT_NAMES: Dict[str, str] = {
    "hypoxemia": "低氧/氧合危机 Agent",
    "ventilation_etco2": "通气与 EtCO2 异常 Agent",
    "hypotension": "低血压/低灌注 Agent",
    "hypertension": "高血压/交感反应 Agent",
    "depth_of_anesthesia": "麻醉深度异常 Agent",
    "bradycardia": "心动过缓 Agent",
    "tachycardia": "心动过速 Agent",
    "hemorrhage_or_volume": "出血与容量管理 Agent",
    "general_warning": "一般预警 Agent",
}


class PerioperativeCrisisState(TypedDict, total=False):
    case: Dict[str, Any]
    snapshot: Dict[str, Any]
    config: Dict[str, Any]
    detected_events: List[Dict[str, Any]]
    primary_event: str
    severity: str
    route_reason: str
    agent_outputs: Dict[str, Dict[str, Any]]
    safety_alerts: List[str]
    missing_information: List[str]
    audit_trail: List[Dict[str, Any]]
    final_report: Dict[str, Any]
    human_review_required: bool


@dataclass
class CrisisAgentOutput:
    agent: str
    event_type: str
    severity: str
    assessment: str
    immediate_checks: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    safety_constraints: List[str] = field(default_factory=list)
    reassessment_plan: List[str] = field(default_factory=list)
    evidence_notes: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    handoff: str = ""
    confidence: str = "medium"

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
    if not isinstance(tags, list):
        return []
    return [str(x) for x in tags]


def _risk_flags(snapshot: Dict[str, Any]) -> List[str]:
    flags = _clinical_assessment(snapshot).get("risk_flags", [])
    if not isinstance(flags, list):
        return []
    return [str(x) for x in flags]


def _event(event_type: str, severity: str, evidence: str) -> Dict[str, Any]:
    return {"event_type": event_type, "severity": severity, "evidence": evidence}


def _contextual_severity(snapshot: Dict[str, Any], default: str) -> str:
    category = str(snapshot.get("sample_category", "")).lower()
    risk_level = str(_clinical_assessment(snapshot).get("risk_level", "")).lower()
    if category in {"critical_alarm", "critical"} or risk_level in {"critical", "high"}:
        return "critical"
    return default


def normalize_dataset_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Accept a dataset row or a raw snapshot and return a workflow input case."""
    if "snapshot" in row and isinstance(row["snapshot"], dict):
        snapshot = deepcopy(row["snapshot"])
        case = {
            "caseid": row.get("caseid"),
            "surgery_group": row.get("surgery_group") or snapshot.get("surgery_group"),
            "generation_mode": row.get("generation_mode"),
            "sample_category": snapshot.get("sample_category"),
            "miller_alignment": snapshot.get("miller_alignment", {}),
            "actual_intervention": snapshot.get("actual_intervention") or row.get("actual_intervention"),
        }
        return {"case": case, "snapshot": snapshot}
    return {"case": {}, "snapshot": deepcopy(row)}


class PerioperativeCrisisWorkflow:
    """LangGraph-compatible event-driven crisis workflow."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(config or {})

    def initial_state(self, row_or_snapshot: Dict[str, Any]) -> PerioperativeCrisisState:
        normalized = normalize_dataset_row(row_or_snapshot)
        return {
            "case": normalized["case"],
            "snapshot": normalized["snapshot"],
            "config": dict(self.config),
            "detected_events": [],
            "primary_event": "",
            "severity": "stable",
            "route_reason": "",
            "agent_outputs": {},
            "safety_alerts": [],
            "missing_information": [],
            "audit_trail": [],
            "final_report": {},
            "human_review_required": False,
        }

    def invoke(self, row_or_snapshot: Dict[str, Any]) -> PerioperativeCrisisState:
        app = self.compile()
        return app.invoke(self.initial_state(row_or_snapshot))

    def compile(self) -> Any:
        try:
            from langgraph.graph import END, START, StateGraph  # type: ignore

            graph = StateGraph(PerioperativeCrisisState)
            graph.add_node("snapshot_intake", self.snapshot_intake)
            graph.add_node("anomaly_detector", self.anomaly_detector)
            graph.add_node("triage_router", self.triage_router)
            for event_type in CRISIS_EVENT_ORDER:
                graph.add_node(event_type, self._event_node(event_type))
                graph.add_edge(event_type, "evidence_reviewer")
            graph.add_node("evidence_reviewer", self.evidence_reviewer)
            graph.add_node("safety_gate", self.safety_gate)
            graph.add_edge(START, "snapshot_intake")
            graph.add_edge("snapshot_intake", "anomaly_detector")
            graph.add_edge("anomaly_detector", "triage_router")
            graph.add_conditional_edges("triage_router", self._route_event, {x: x for x in CRISIS_EVENT_ORDER})
            graph.add_edge("evidence_reviewer", "safety_gate")
            graph.add_edge("safety_gate", END)
            return graph.compile()
        except Exception:
            return _SequentialCrisisApp(self)

    def snapshot_intake(self, state: PerioperativeCrisisState) -> PerioperativeCrisisState:
        next_state = deepcopy(state)
        snapshot = next_state.get("snapshot", {})
        missing = []
        for key in ("patient_background", "intraop_stage", "vital_stats", "clinical_assessment"):
            if key not in snapshot:
                missing.append(f"snapshot 缺少字段: {key}")
        next_state["missing_information"] = _dedupe(list(next_state.get("missing_information", [])) + missing)
        next_state["audit_trail"] = list(next_state.get("audit_trail", [])) + [
            {
                "stage": "snapshot_intake",
                "caseid": next_state.get("case", {}).get("caseid"),
                "missing_information": missing,
            }
        ]
        if missing:
            next_state["human_review_required"] = True
        return next_state

    def anomaly_detector(self, state: PerioperativeCrisisState) -> PerioperativeCrisisState:
        next_state = deepcopy(state)
        snapshot = next_state.get("snapshot", {})
        recent = _recent(snapshot)
        persist = _persistence(snapshot)
        alarms = set(_alarm_tags(snapshot))
        events: List[Dict[str, Any]] = []

        spo2 = _as_float(recent.get("SpO2_pct"))
        map_v = _as_float(recent.get("MAP_mmhg"))
        sbp = _as_float(recent.get("SBP_mmhg"))
        dbp = _as_float(recent.get("DBP_mmhg"))
        hr = _as_float(recent.get("HR_bpm"))
        etco2 = _as_float(recent.get("EtCO2_mmhg"))
        bis = _as_float(recent.get("BIS"))
        svv = _as_float(recent.get("SVV_pct"))

        if spo2 is not None and spo2 < 90:
            events.append(_event("hypoxemia", "critical", f"SpO2={spo2:.1f}%"))
        elif spo2 is not None and (spo2 < 94 or "SpO2" in alarms):
            events.append(_event("hypoxemia", "warning", f"SpO2={spo2:.1f}% or SpO2 alarm"))

        etco2_missing = _as_float(persist.get("etco2_missing")) or 0.0
        etco2_zero = _as_float(persist.get("etco2_zero_like")) or 0.0
        if "EtCO2" in alarms or etco2_missing > 0 or etco2_zero > 0:
            severity = "critical" if etco2_zero >= 10 or etco2_missing >= 10 else "warning"
            events.append(
                _event(
                    "ventilation_etco2",
                    _contextual_severity(snapshot, severity),
                    f"EtCO2={etco2}; missing={etco2_missing}s zero={etco2_zero}s",
                )
            )
        elif etco2 is not None and (etco2 < 25 or etco2 > 55):
            events.append(_event("ventilation_etco2", "warning", f"EtCO2={etco2:.1f} mmHg"))

        if map_v is not None and map_v < 55:
            events.append(_event("hypotension", "critical", f"MAP={map_v:.1f} mmHg"))
        elif map_v is not None and (map_v < 65 or "MAP" in alarms):
            events.append(_event("hypotension", "warning", f"MAP={map_v:.1f} mmHg or MAP alarm"))

        if (sbp is not None and sbp > 180) or (dbp is not None and dbp > 100):
            events.append(_event("hypertension", "warning", f"SBP={sbp}, DBP={dbp}"))
        elif "SBP" in alarms or "DBP" in alarms:
            events.append(_event("hypertension", _contextual_severity(snapshot, "warning"), "relative SBP/DBP alarm"))

        if bis is not None and bis > 60:
            events.append(_event("depth_of_anesthesia", _contextual_severity(snapshot, "warning"), f"BIS={bis:.1f}"))
        elif "BIS" in alarms:
            events.append(
                _event(
                    "depth_of_anesthesia",
                    _contextual_severity(snapshot, "warning"),
                    "BIS alarm or missing-depth concern",
                )
            )

        if hr is not None and hr < 45:
            events.append(_event("bradycardia", "critical", f"HR={hr:.1f} bpm"))
        elif hr is not None and hr < 50:
            events.append(_event("bradycardia", "warning", f"HR={hr:.1f} bpm"))
        if hr is not None and hr > 120:
            events.append(_event("tachycardia", "critical", f"HR={hr:.1f} bpm"))
        elif hr is not None and hr > 100:
            events.append(_event("tachycardia", "warning", f"HR={hr:.1f} bpm"))

        if svv is not None and svv >= 13:
            events.append(_event("hemorrhage_or_volume", "warning", f"SVV={svv:.1f}%"))

        if not events:
            risk_level = str(_clinical_assessment(snapshot).get("risk_level", "stable"))
            category = str(snapshot.get("sample_category", "stable"))
            events.append(_event("general_warning", risk_level if risk_level else "stable", f"sample_category={category}"))

        next_state["detected_events"] = events
        next_state["audit_trail"] = list(next_state.get("audit_trail", [])) + [
            {"stage": "anomaly_detector", "events": events, "alarm_tags": list(alarms)}
        ]
        return next_state

    def triage_router(self, state: PerioperativeCrisisState) -> PerioperativeCrisisState:
        next_state = deepcopy(state)
        events = list(next_state.get("detected_events", []))
        severity_rank = {"critical": 3, "high": 3, "warning": 2, "moderate": 2, "stable": 1, "low": 1}
        priority = {event: i for i, event in enumerate(CRISIS_EVENT_ORDER)}
        primary = sorted(
            events,
            key=lambda e: (
                -severity_rank.get(str(e.get("severity", "stable")).lower(), 1),
                priority.get(str(e.get("event_type", "general_warning")), 999),
            ),
        )[0]
        event_type = str(primary.get("event_type", "general_warning"))
        if event_type not in CRISIS_EVENT_ORDER:
            event_type = "general_warning"
        next_state["primary_event"] = event_type
        next_state["severity"] = str(primary.get("severity", "stable"))
        next_state["route_reason"] = str(primary.get("evidence", ""))
        next_state["audit_trail"] = list(next_state.get("audit_trail", [])) + [
            {
                "stage": "triage_router",
                "primary_event": event_type,
                "severity": next_state["severity"],
                "reason": next_state["route_reason"],
            }
        ]
        if next_state["severity"] in {"critical", "high", "warning", "moderate"}:
            next_state["human_review_required"] = True
        return next_state

    def evidence_reviewer(self, state: PerioperativeCrisisState) -> PerioperativeCrisisState:
        next_state = deepcopy(state)
        snapshot = next_state.get("snapshot", {})
        alignment = snapshot.get("miller_alignment", {})
        notes = []
        alerts = []
        if isinstance(alignment, dict):
            verdict = str(alignment.get("verdict", ""))
            reason = str(alignment.get("reason", ""))
            if verdict:
                notes.append(f"Miller/VitalDB alignment verdict: {verdict}")
            if reason:
                notes.append(f"Alignment reason: {reason}")
            if bool(alignment.get("high_risk_conflict", False)):
                alerts.append("数据集标记为 high_risk_conflict，需要人工复核")
        retrieval = snapshot.get("miller_retrieval") or next_state.get("case", {}).get("miller_retrieval")
        if retrieval:
            notes.append("存在 Miller 文献检索结果，可用于 RAG 证据核查")
        next_state["safety_alerts"] = _dedupe(list(next_state.get("safety_alerts", [])) + alerts)
        outputs = dict(next_state.get("agent_outputs", {}))
        key = next_state.get("primary_event", "general_warning")
        if key in outputs:
            old = dict(outputs[key])
            old["evidence_notes"] = _dedupe(list(old.get("evidence_notes", [])) + notes)
            outputs[key] = old
        next_state["agent_outputs"] = outputs
        next_state["audit_trail"] = list(next_state.get("audit_trail", [])) + [
            {"stage": "evidence_reviewer", "notes": notes, "alerts": alerts}
        ]
        if alerts:
            next_state["human_review_required"] = True
        return next_state

    def safety_gate(self, state: PerioperativeCrisisState) -> PerioperativeCrisisState:
        next_state = deepcopy(state)
        snapshot = next_state.get("snapshot", {})
        primary_event = next_state.get("primary_event", "general_warning")
        safety_alerts = list(next_state.get("safety_alerts", []))
        severity = str(next_state.get("severity", "stable"))
        if severity in {"critical", "high"}:
            safety_alerts.append("危急级别事件：必须由麻醉医生立即复核，Agent 不可自动执行处置")
        if primary_event in {"hypotension", "hypoxemia", "ventilation_etco2", "bradycardia"}:
            safety_alerts.append("优先确认监测信号、气道/通气和循环灌注，避免直接按单一数值调整药物")
        if snapshot.get("waveform_image_path"):
            safety_alerts.append("可结合 waveform image 进行二次视觉核查")
        next_state["safety_alerts"] = _dedupe(safety_alerts)
        next_state["human_review_required"] = bool(
            next_state.get("human_review_required", False) or next_state["safety_alerts"]
        )
        final_report = {
            "caseid": next_state.get("case", {}).get("caseid"),
            "workflow": "perioperative_crisis_anesthesia_agent",
            "framework": "LangGraph orchestration with deterministic fallback",
            "primary_event": primary_event,
            "severity": severity,
            "route_reason": next_state.get("route_reason", ""),
            "detected_events": next_state.get("detected_events", []),
            "agent_outputs": next_state.get("agent_outputs", {}),
            "safety_alerts": next_state["safety_alerts"],
            "missing_information": next_state.get("missing_information", []),
            "human_review_required": next_state["human_review_required"],
            "clinical_safety_notice": (
                "本报告仅用于围术期麻醉突发情况决策支持、训练评估和质控；不得作为自动医嘱或自动给药依据。"
                "所有处置必须由具备资质的麻醉医生结合患者实时状态、设备、药品说明书和本院规范确认。"
            ),
        }
        next_state["final_report"] = final_report
        next_state["audit_trail"] = list(next_state.get("audit_trail", [])) + [
            {"stage": "safety_gate", "human_review_required": next_state["human_review_required"]}
        ]
        return next_state

    def _route_event(self, state: PerioperativeCrisisState) -> str:
        event_type = str(state.get("primary_event", "general_warning"))
        return event_type if event_type in CRISIS_EVENT_ORDER else "general_warning"

    def _event_node(self, event_type: str) -> Callable[[PerioperativeCrisisState], PerioperativeCrisisState]:
        def node(state: PerioperativeCrisisState) -> PerioperativeCrisisState:
            output = self._build_event_output(event_type, state)
            next_state = deepcopy(state)
            outputs = dict(next_state.get("agent_outputs", {}))
            outputs[event_type] = output.to_dict()
            next_state["agent_outputs"] = outputs
            next_state["missing_information"] = _dedupe(
                list(next_state.get("missing_information", [])) + output.missing_information
            )
            next_state["safety_alerts"] = _dedupe(
                list(next_state.get("safety_alerts", [])) + output.safety_constraints
            )
            next_state["audit_trail"] = list(next_state.get("audit_trail", [])) + [
                {
                    "stage": event_type,
                    "agent": output.agent,
                    "severity": output.severity,
                    "confidence": output.confidence,
                }
            ]
            return next_state

        return node

    def _build_event_output(self, event_type: str, state: PerioperativeCrisisState) -> CrisisAgentOutput:
        snapshot = state.get("snapshot", {})
        recent = _recent(snapshot)
        trends = snapshot.get("vital_trend_last_5min", {})
        risk_flags = _risk_flags(snapshot)
        severity = str(state.get("severity", "warning"))
        route_reason = str(state.get("route_reason", ""))
        base = {
            "agent": EVENT_AGENT_NAMES.get(event_type, "围术期事件 Agent"),
            "event_type": event_type,
            "severity": severity,
            "evidence_notes": risk_flags[:5],
            "confidence": "medium",
        }
        if event_type == "hypoxemia":
            return CrisisAgentOutput(
                **base,
                assessment=f"识别到氧合风险：{route_reason}",
                immediate_checks=[
                    "确认 SpO2 波形质量、探头位置和灌注状态",
                    "提高氧合支持并确认当前 FiO2、气道压力和潮气量",
                    "胸科/单肺通气场景下复核双腔管或支气管封堵器位置",
                ],
                recommended_actions=[
                    "按气道-通气-循环顺序排查可逆原因",
                    "必要时暂停手术刺激并请求麻醉上级/外科团队协同",
                    "结合 EtCO2、气道压、听诊/纤支镜和 ABG 复核低氧原因",
                ],
                safety_constraints=[
                    "低氧优先级高于单纯加深麻醉或降压处理",
                    "不得仅凭 SpO2 单点值自动给药或改变通气策略",
                ],
                reassessment_plan=["1 分钟内复核 SpO2/EtCO2/气道压", "3-5 分钟内评估干预后趋势和 ABG 需求"],
                handoff="若合并 EtCO2 消失或血压下降，转入通气/循环联合危机处理。",
            )
        if event_type == "ventilation_etco2":
            return CrisisAgentOutput(
                **base,
                assessment=f"识别到 EtCO2/通气监测异常：{route_reason}",
                immediate_checks=[
                    "先确认 EtCO2 采样管、接头、过滤器和呼吸回路是否脱落/堵塞/漏气",
                    "查看胸廓起伏、呼吸机参数、气道压和听诊结果",
                    "胸科病例需复核单肺通气器械位置和肺隔离状态",
                ],
                recommended_actions=[
                    "若 SpO2 尚稳定，优先区分监测故障与真实通气中断",
                    "若 EtCO2 持续为 0 或合并低氧，按气道危机升级处理",
                    "同步评估麻醉深度和循环，避免在通气未确认前盲目加深麻醉",
                ],
                safety_constraints=[
                    "EtCO2 消失是高风险信号，必须人工确认气道和呼吸回路",
                    "采样故障未排除前，不应把 EtCO2=0 简单解释为代谢下降",
                ],
                reassessment_plan=["立即复核 EtCO2 波形", "1 分钟内记录回路检查结果", "5 分钟内复核 SpO2、HR、MAP 趋势"],
                handoff="若同时 BIS 高，交由麻醉深度 Agent 在通气安全确认后处理。",
            )
        if event_type == "hypotension":
            return CrisisAgentOutput(
                **base,
                assessment=f"识别到低血压/低灌注风险：{route_reason}",
                immediate_checks=[
                    "确认动脉压/NIBP 信号可靠性和袖带/换能器位置",
                    "快速评估出血、容量、麻醉深度、心率和过敏/气胸等可逆原因",
                    "查看尿量、SVV/PPV/CVP/CO/CI 等可用灌注指标",
                ],
                recommended_actions=[
                    "优先恢复有效灌注，必要时请求上级麻醉医生和外科暂停刺激",
                    "根据容量反应性、心率和临床背景选择补液/血管活性药策略",
                    "若合并低氧、EtCO2 异常或大出血，进入复合危机路径",
                ],
                safety_constraints=[
                    "MAP < 65 或快速下降时，不应优先加深丙泊酚/阿片类药物",
                    "血管活性药和补液策略必须由临床医生确认",
                ],
                reassessment_plan=["30-60 秒复核 MAP/HR", "3 分钟内评估灌注改善", "记录干预前后血流动力学变化"],
                handoff="若与用药调整冲突，交由安全审核 Agent 标记 high-risk conflict。",
            )
        if event_type == "hypertension":
            return CrisisAgentOutput(
                **base,
                assessment=f"识别到高血压/交感反应风险：{route_reason}",
                immediate_checks=[
                    "确认血压测量可靠性，排除动脉波形阻尼/冲洗/体位问题",
                    "评估手术刺激、镇痛不足、麻醉过浅、低氧/高碳酸血症和膀胱充盈等原因",
                    "结合 BIS、HR、EtCO2 和手术阶段判断是疼痛/觉醒风险还是其他病因",
                ],
                recommended_actions=[
                    "优先处理可逆诱因，再考虑镇痛/镇静或降压策略",
                    "胸科手术中同步关注单肺通气导致的氧合/CO2 变化",
                    "若 MAP 接近低灌注边界，避免过度降压",
                ],
                safety_constraints=["降压或加深麻醉前需确认没有低氧、低灌注或通气异常"],
                reassessment_plan=["1-3 分钟复核 BP/HR/BIS", "记录手术刺激变化和处理反应"],
                handoff="若 BIS 高，联动麻醉深度 Agent；若 EtCO2 异常，先处理通气。",
            )
        if event_type == "depth_of_anesthesia":
            return CrisisAgentOutput(
                **base,
                assessment=f"识别到麻醉深度异常或 BIS 相关风险：{route_reason}",
                immediate_checks=[
                    "确认 BIS 电极质量、肌电干扰和信号质量",
                    "结合体动、HR/BP、手术刺激和输注泵状态判断麻醉深度",
                    "复核丙泊酚/瑞芬太尼等输注通路、泵速单位和 TCI 状态",
                ],
                recommended_actions=[
                    "若通气和循环稳定，可考虑在医生确认下逐步调整镇静/镇痛",
                    "若合并低血压或低氧，优先处理灌注/氧合，不应单纯加深麻醉",
                    "用小步调整和短周期复评替代大幅度一次性改变",
                ],
                safety_constraints=["BIS 只能作为支持信号，不能替代临床综合判断"],
                reassessment_plan=["1 分钟复核 BIS 趋势和生命体征", "3-5 分钟复核是否进入目标范围"],
                handoff="若数据集 Miller alignment 标记冲突，交由证据审核节点复核。",
            )
        if event_type == "bradycardia":
            return CrisisAgentOutput(
                **base,
                assessment=f"识别到心动过缓风险：{route_reason}",
                immediate_checks=["确认 ECG/脉搏波一致性", "评估血压、麻醉深度、迷走刺激、缺氧和药物因素"],
                recommended_actions=["若合并低血压或低灌注，按症状性心动过缓升级处理", "请临床医生确认抗胆碱药/升压策略"],
                safety_constraints=["稳定轻度低心率不等同于必须立即给药"],
                reassessment_plan=["30-60 秒复核 HR/MAP", "记录是否存在手术牵拉或药物触发"],
                handoff="若合并低血压，联动低血压 Agent。",
            )
        if event_type == "tachycardia":
            return CrisisAgentOutput(
                **base,
                assessment=f"识别到心动过速风险：{route_reason}",
                immediate_checks=["确认 ECG 节律类型", "评估疼痛/麻醉浅、低容量、低氧、高碳酸血症和发热"],
                recommended_actions=["优先处理诱因，必要时请求人工确认抗心律失常或镇痛镇静策略"],
                safety_constraints=["未明确节律和病因前不应自动推荐特定抗心律失常药"],
                reassessment_plan=["1 分钟复核 HR/MAP/SpO2/EtCO2", "必要时获取 12 导联或血气/电解质"],
                handoff="若伴随血压不稳，进入循环危机处理。",
            )
        if event_type == "hemorrhage_or_volume":
            return CrisisAgentOutput(
                **base,
                assessment=f"识别到容量/出血相关风险：{route_reason}",
                immediate_checks=["核对术野出血、吸引量、纱布和尿量", "查看 Hb/ABG、SVV/PPV、MAP/HR 和血管活性药需求"],
                recommended_actions=["与外科确认出血控制", "按本院输血和容量管理流程准备液体/血制品", "动态复评容量反应性"],
                safety_constraints=["未确认容量状态前避免单纯升压掩盖低容量"],
                reassessment_plan=["3-5 分钟复核 MAP/HR/SVV/尿量", "必要时复查血气和 Hb"],
                handoff="若 MAP 下降，联动低血压 Agent。",
            )
        return CrisisAgentOutput(
            **base,
            assessment=f"未进入单一危机路径，当前为一般预警：{route_reason}",
            immediate_checks=["复核监测信号质量", "查看近 5-10 分钟趋势和手术阶段"],
            recommended_actions=["保持连续监测，小步可逆调整", "补齐缺失指标后再次分诊"],
            safety_constraints=["一般预警也需保留人工复核和审计记录"],
            reassessment_plan=["3-5 分钟后复核趋势"],
            evidence_notes=[json.dumps(trends, ensure_ascii=False)[:500]],
            handoff="若出现新报警，重新进入 anomaly_detector。",
        )


class _SequentialCrisisApp:
    def __init__(self, workflow: PerioperativeCrisisWorkflow) -> None:
        self.workflow = workflow

    def invoke(self, state: PerioperativeCrisisState) -> PerioperativeCrisisState:
        state = self.workflow.snapshot_intake(state)
        state = self.workflow.anomaly_detector(state)
        state = self.workflow.triage_router(state)
        route = self.workflow._route_event(state)
        state = self.workflow._event_node(route)(state)
        state = self.workflow.evidence_reviewer(state)
        state = self.workflow.safety_gate(state)
        return state


def run_perioperative_crisis_workflow(
    row_or_snapshot: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    workflow = PerioperativeCrisisWorkflow(config=config)
    state = workflow.invoke(row_or_snapshot)
    report = state.get("final_report", {})
    report["audit_trail"] = state.get("audit_trail", [])
    return report
