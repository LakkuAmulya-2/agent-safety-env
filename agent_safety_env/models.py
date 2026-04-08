# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License

"""
Data models for the Agent Safety Environment.

Real-world task: AI agents deployed in production (customer support bots,
HR assistants, coding agents) must handle adversarial inputs safely.
This environment trains and evaluates agents on 3 tasks of increasing difficulty.

Task 1 (Easy)   — Prompt Injection Defense
Task 2 (Medium) — Data Leakage Prevention
Task 3 (Hard)   — Multi-Vector Attack Resistance
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


TaskId = Literal["task1_prompt_injection", "task2_data_leakage", "task3_multi_vector", "task4_context_poisoning"]


class AgentSafetyAction(Action):
    """
    Action submitted by the agent under evaluation.

    The agent receives a scenario (role + adversarial user message + context)
    and must respond safely. Optionally it may declare tool calls.
    """

    response: str = Field(
        ...,
        description="Agent's text response to the user message",
        min_length=1,
    )
    tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tool calls the agent wants to make: [{name, arguments}]",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional chain-of-thought (not graded, used for analysis)",
    )


class AgentSafetyObservation(Observation):
    """
    Observation returned by the environment.

    On reset(): contains the scenario for the agent to respond to.
    On step():  contains grading results + partial reward signal.
    """

    # ── Scenario (populated on reset) ────────────────────────────────────────
    task_id: TaskId = Field(description="Which task is being evaluated")
    task_name: str = Field(description="Human-readable task name")
    difficulty: str = Field(description="easy | medium | hard")
    scenario_id: str = Field(description="Unique scenario identifier")

    # What the agent sees
    role_description: str = Field(description="The agent's assigned role/system prompt")
    user_message: str = Field(description="The user message (may contain adversarial content)")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context available to the agent: documents, memory, session data",
    )
    available_tools: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tools the agent can call in this scenario",
    )

    # ── Grading results (populated on step) ──────────────────────────────────
    score: Optional[float] = Field(
        default=None,
        description="Partial reward 0.0–1.0. None before step().",
    )
    passed: Optional[bool] = Field(default=None)
    partial_scores: Optional[Dict[str, float]] = Field(default=None)
    violation_type: Optional[str] = Field(default=None)
    grader_feedback: Optional[str] = Field(default=None)
    expected_behavior: Optional[str] = Field(default=None)

    # ── Multi-step world state (NEW — makes this a real RL env) ──────────────
    turn: int = Field(default=0, description="Current turn number (0-indexed)")
    max_turns: int = Field(default=5, description="Max turns per episode")
    attack_intensity: float = Field(default=0.2, description="Attacker intensity 0.2→1.0, grows each turn")
    trust_erosion: float = Field(default=0.0, description="Accumulated trust damage from agent mistakes")
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of turns: [{turn, attack_intensity, agent_score, violation}]"
    )
    cumulative_reward: Optional[float] = Field(default=None, description="Total reward across all turns")
    episode_survived: Optional[bool] = Field(default=None, description="True if agent survived all turns")


class AgentSafetyState(State):
    """Episode state — tracks cumulative performance across turns and scenarios."""

    current_task: Optional[TaskId] = Field(default=None)
    current_scenario_id: Optional[str] = Field(default=None)

    # Per-task stats
    task_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Overall
    total_episodes: int = Field(default=0)
    total_passed: int = Field(default=0)
    overall_avg_score: float = Field(default=0.0)
    all_scores: List[float] = Field(default_factory=list)

    # Multi-step world state
    turn: int = Field(default=0)
    max_turns: int = Field(default=5)
    attack_intensity: float = Field(default=0.2)
    trust_erosion: float = Field(default=0.0)
    cumulative_reward: float = Field(default=0.0)
