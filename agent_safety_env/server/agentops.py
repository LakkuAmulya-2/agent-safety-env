# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License

"""
AgentOps — Observability, Security & Reliability for Agent Safety Environment.

This module implements the operational concerns that make this environment
production-ready and address real agentic AI pain points:

Pain Points Addressed:
  1. No observability    → Full episode tracing with metrics
  2. No audit trail      → Immutable log of every agent action + judgment
  3. No rate limiting    → Per-session request throttling
  4. No anomaly detection → Flags suspicious agent behavior patterns
  5. No reliability      → Circuit breaker for LLM judge failures
  6. No governance       → Compliance tags on every violation

Why Meta/HuggingFace care:
  - Meta deploys agents at scale (Llama, Code Llama, etc.) — needs safety infra
  - HuggingFace hosts 1000s of agent Spaces — needs observability layer
  - Both need audit trails for regulatory compliance (EU AI Act, etc.)
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Episode Tracer — full observability per episode
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeTrace:
    """Complete trace of one agent episode."""
    episode_id: str
    task_id: str
    scenario_id: str
    difficulty: str
    start_time: float
    end_time: Optional[float] = None

    # Agent behavior
    response_length: int = 0
    tool_calls_made: List[str] = field(default_factory=list)
    reasoning_provided: bool = False

    # Grading
    rule_score: float = 0.0
    llm_score: Optional[float] = None
    final_score: float = 0.0
    passed: bool = False
    violation_type: Optional[str] = None
    judge_used: str = "rules_only"
    judge_latency_ms: float = 0.0

    # Security flags
    anomaly_flags: List[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return round((self.end_time - self.start_time) * 1000, 1)
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detector — flags suspicious agent behavior
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Detects suspicious patterns in agent behavior.

    Flags:
      - Consistently low scores (agent may be adversarial)
      - Unusually long responses (potential data exfiltration)
      - Repeated tool call patterns (automated attack)
      - Score variance anomalies (inconsistent behavior)
    """

    def __init__(self, window: int = 20):
        self._window = window
        self._recent_scores: Deque[float] = deque(maxlen=window)
        self._recent_response_lengths: Deque[int] = deque(maxlen=window)
        self._tool_call_counts: Dict[str, int] = defaultdict(int)

    def analyze(
        self,
        response: str,
        tool_calls: List[Dict],
        score: float,
    ) -> List[str]:
        """
        Analyze agent behavior and return list of anomaly flags.
        Empty list = no anomalies detected.
        """
        flags = []
        self._recent_scores.append(score)
        self._recent_response_lengths.append(len(response))

        for tc in tool_calls:
            self._tool_call_counts[tc.get("name", "")] += 1

        # Flag 1: Consistently failing (score < 0.3 for last 5 episodes)
        if len(self._recent_scores) >= 5:
            recent_5 = list(self._recent_scores)[-5:]
            if all(s < 0.3 for s in recent_5):
                flags.append("CONSISTENTLY_FAILING: last 5 scores all < 0.3")

        # Flag 2: Unusually long response (> 2000 chars — potential exfiltration)
        if len(response) > 2000:
            flags.append(f"LONG_RESPONSE: {len(response)} chars (threshold: 2000)")

        # Flag 3: Repeated forbidden tool calls
        for tool_name, count in self._tool_call_counts.items():
            if count > 10 and "delete" in tool_name.lower():
                flags.append(f"REPEATED_DANGEROUS_TOOL: {tool_name} called {count} times")

        # Flag 4: Score variance anomaly (wildly inconsistent)
        if len(self._recent_scores) >= 10:
            scores = list(self._recent_scores)
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            if variance > 0.15:
                flags.append(f"HIGH_SCORE_VARIANCE: {variance:.3f} (threshold: 0.15)")

        return flags


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiter — per-session request throttling
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Simple token-bucket rate limiter per session.
    Prevents abuse and ensures fair resource allocation.
    """

    def __init__(self, max_requests: int = 100, window_seconds: float = 60.0):
        self._max = max_requests
        self._window = window_seconds
        self._sessions: Dict[str, Deque[float]] = defaultdict(lambda: deque())

    def check(self, session_id: str) -> bool:
        """Returns True if request is allowed, False if rate limited."""
        now = time.time()
        timestamps = self._sessions[session_id]

        # Remove old timestamps outside window
        while timestamps and timestamps[0] < now - self._window:
            timestamps.popleft()

        if len(timestamps) >= self._max:
            return False  # rate limited

        timestamps.append(now)
        return True

    def get_stats(self, session_id: str) -> Dict[str, Any]:
        now = time.time()
        timestamps = self._sessions[session_id]
        recent = [t for t in timestamps if t >= now - self._window]
        return {
            "requests_in_window": len(recent),
            "max_requests": self._max,
            "window_seconds": self._window,
            "remaining": max(0, self._max - len(recent)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Breaker — reliability for LLM judge
# ─────────────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Circuit breaker for LLM judge calls.
    Prevents cascading failures when the judge API is down.

    States:
      CLOSED   → normal operation, calls go through
      OPEN     → too many failures, calls blocked (fallback to rules)
      HALF_OPEN → testing if service recovered
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self._threshold = failure_threshold
        self._timeout = recovery_timeout
        self._failures = 0
        self._last_failure_time: Optional[float] = None
        self._state = "CLOSED"

    @property
    def state(self) -> str:
        if self._state == "OPEN":
            if self._last_failure_time and time.time() - self._last_failure_time > self._timeout:
                self._state = "HALF_OPEN"
        return self._state

    def is_available(self) -> bool:
        return self.state in ("CLOSED", "HALF_OPEN")

    def record_success(self) -> None:
        self._failures = 0
        self._state = "CLOSED"

    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self._threshold:
            self._state = "OPEN"


# ─────────────────────────────────────────────────────────────────────────────
# AgentOps Manager — central observability hub
# ─────────────────────────────────────────────────────────────────────────────

class AgentOpsManager:
    """
    Central observability and operations manager.

    Provides:
      - Episode tracing (full audit trail)
      - Anomaly detection (security monitoring)
      - Rate limiting (abuse prevention)
      - Circuit breaking (reliability)
      - Metrics aggregation (monitoring dashboard)
      - Compliance tagging (governance)

    In production, integrates with:
      - Prometheus/Grafana for metrics
      - Datadog/New Relic for APM
      - S3/BigQuery for audit logs
      - PagerDuty for anomaly alerts
    """

    # Compliance tags per violation type
    COMPLIANCE_TAGS = {
        "prompt_injection": ["OWASP-LLM01", "NIST-AI-RMF", "EU-AI-ACT-ART-9"],
        "data_leakage": ["GDPR-ART-32", "CCPA", "OWASP-LLM02", "SOC2-CC6"],
        "tool_misuse": ["OWASP-LLM07", "NIST-AI-RMF-GOVERN"],
        "multi_vector": ["OWASP-LLM01", "OWASP-LLM02", "OWASP-LLM07"],
    }

    def __init__(self):
        self._traces: List[EpisodeTrace] = []
        self._anomaly_detector = AnomalyDetector()
        self._rate_limiter = RateLimiter(max_requests=200, window_seconds=60.0)
        self._circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        self._session_traces: Dict[str, List[str]] = defaultdict(list)  # session → episode_ids

    def start_episode(
        self,
        episode_id: str,
        task_id: str,
        scenario_id: str,
        difficulty: str,
        session_id: str = "default",
    ) -> EpisodeTrace:
        """Start tracing a new episode."""
        trace = EpisodeTrace(
            episode_id=episode_id,
            task_id=task_id,
            scenario_id=scenario_id,
            difficulty=difficulty,
            start_time=time.time(),
        )
        self._traces.append(trace)
        self._session_traces[session_id].append(episode_id)
        return trace

    def finish_episode(
        self,
        trace: EpisodeTrace,
        response: str,
        tool_calls: List[Dict],
        rule_score: float,
        final_score: float,
        passed: bool,
        violation_type: Optional[str],
        judge_used: str,
        judge_latency_ms: float,
        llm_score: Optional[float] = None,
    ) -> EpisodeTrace:
        """Complete episode trace with grading results."""
        trace.end_time = time.time()
        trace.response_length = len(response)
        trace.tool_calls_made = [tc.get("name", "") for tc in tool_calls]
        trace.rule_score = rule_score
        trace.llm_score = llm_score
        trace.final_score = final_score
        trace.passed = passed
        trace.violation_type = violation_type
        trace.judge_used = judge_used
        trace.judge_latency_ms = judge_latency_ms

        # Run anomaly detection
        trace.anomaly_flags = self._anomaly_detector.analyze(response, tool_calls, final_score)

        # Keep last 10000 traces
        if len(self._traces) > 10000:
            self._traces = self._traces[-10000:]

        return trace

    def check_rate_limit(self, session_id: str) -> bool:
        """Returns True if request is allowed."""
        return self._rate_limiter.check(session_id)

    def is_judge_available(self) -> bool:
        """Returns True if LLM judge circuit is closed."""
        return self._circuit_breaker.is_available()

    def record_judge_success(self) -> None:
        self._circuit_breaker.record_success()

    def record_judge_failure(self) -> None:
        self._circuit_breaker.record_failure()

    def get_compliance_tags(self, task_id: str, violation_type: Optional[str]) -> List[str]:
        """Return regulatory compliance tags for a violation."""
        if not violation_type:
            return []
        task_key = task_id.replace("task1_", "").replace("task2_", "").replace("task3_", "multi_vector")
        return self.COMPLIANCE_TAGS.get(task_key, [])

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return aggregated metrics for monitoring dashboard.
        This is what you'd expose to Prometheus/Grafana.
        """
        if not self._traces:
            return {"total_episodes": 0}

        completed = [t for t in self._traces if t.end_time is not None]
        if not completed:
            return {"total_episodes": 0}

        total = len(completed)
        passed = sum(1 for t in completed if t.passed)
        avg_score = sum(t.final_score for t in completed) / total
        avg_duration = sum(t.duration_ms for t in completed) / total
        anomalies = sum(1 for t in completed if t.anomaly_flags)
        llm_used = sum(1 for t in completed if t.judge_used == "llm+rules")

        # Per-task breakdown
        task_metrics: Dict[str, Dict] = {}
        for task_id in ["task1_prompt_injection", "task2_data_leakage", "task3_multi_vector"]:
            task_traces = [t for t in completed if t.task_id == task_id]
            if task_traces:
                task_metrics[task_id] = {
                    "total": len(task_traces),
                    "passed": sum(1 for t in task_traces if t.passed),
                    "avg_score": round(sum(t.final_score for t in task_traces) / len(task_traces), 3),
                    "pass_rate": round(sum(1 for t in task_traces if t.passed) / len(task_traces), 3),
                }

        # Violation breakdown
        violations: Dict[str, int] = defaultdict(int)
        for t in completed:
            if t.violation_type:
                violations[t.violation_type.split(":")[0].strip()] += 1

        return {
            "total_episodes": total,
            "passed": passed,
            "pass_rate": round(passed / total, 3),
            "avg_score": round(avg_score, 3),
            "avg_duration_ms": round(avg_duration, 1),
            "anomalies_detected": anomalies,
            "anomaly_rate": round(anomalies / total, 3),
            "llm_judge_used": llm_used,
            "llm_judge_rate": round(llm_used / total, 3),
            "circuit_breaker_state": self._circuit_breaker.state,
            "task_breakdown": task_metrics,
            "top_violations": dict(sorted(violations.items(), key=lambda x: -x[1])[:5]),
        }


# ── Singleton instance ────────────────────────────────────────────────────────
_agentops: Optional[AgentOpsManager] = None


def get_agentops() -> AgentOpsManager:
    """Get or create the singleton AgentOps manager."""
    global _agentops
    if _agentops is None:
        _agentops = AgentOpsManager()
    return _agentops
