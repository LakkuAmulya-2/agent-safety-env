# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License

"""Agent Safety Environment — OpenEnv Client."""

from typing import Any, Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import AgentSafetyAction, AgentSafetyObservation, AgentSafetyState
except ImportError:
    from models import AgentSafetyAction, AgentSafetyObservation, AgentSafetyState


class AgentSafetyEnv(EnvClient[AgentSafetyAction, AgentSafetyObservation, AgentSafetyState]):
    """
    Client for the Agent Safety Environment.

    Works with any OpenAI-compatible LLM agent.

    Example (sync):
        with AgentSafetyEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset(task_id="task1_prompt_injection")
            obs = result.observation
            print(obs.user_message)

            action = AgentSafetyAction(
                response="I cannot follow those instructions.",
                tool_calls=[],
            )
            result = env.step(action)
            print(result.observation.score)        # 0.0–1.0
            print(result.observation.partial_scores)
            print(result.observation.grader_feedback)

    Example (async):
        async with AgentSafetyEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(task_id="task2_data_leakage")
            result = await env.step(AgentSafetyAction(response="..."))
    """

    def _step_payload(self, action: AgentSafetyAction) -> Dict[str, Any]:
        return {
            "response": action.response,
            "tool_calls": action.tool_calls,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AgentSafetyObservation]:
        obs_data = payload.get("observation", {})
        observation = AgentSafetyObservation(
            task_id=obs_data.get("task_id", "task1_prompt_injection"),
            task_name=obs_data.get("task_name", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            scenario_id=obs_data.get("scenario_id", ""),
            role_description=obs_data.get("role_description", ""),
            user_message=obs_data.get("user_message", ""),
            context=obs_data.get("context", {}),
            available_tools=obs_data.get("available_tools", []),
            score=obs_data.get("score"),
            passed=obs_data.get("passed"),
            partial_scores=obs_data.get("partial_scores"),
            violation_type=obs_data.get("violation_type"),
            grader_feedback=obs_data.get("grader_feedback"),
            expected_behavior=obs_data.get("expected_behavior"),
            # Multi-step world state
            turn=obs_data.get("turn", 0),
            max_turns=obs_data.get("max_turns", 5),
            attack_intensity=obs_data.get("attack_intensity", 0.2),
            trust_erosion=obs_data.get("trust_erosion", 0.0),
            conversation_history=obs_data.get("conversation_history", []),
            cumulative_reward=obs_data.get("cumulative_reward"),
            episode_survived=obs_data.get("episode_survived"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> AgentSafetyState:
        return AgentSafetyState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_task=payload.get("current_task"),
            current_scenario_id=payload.get("current_scenario_id"),
            task_stats=payload.get("task_stats", {}),
            total_episodes=payload.get("total_episodes", 0),
            total_passed=payload.get("total_passed", 0),
            overall_avg_score=payload.get("overall_avg_score", 0.0),
            all_scores=payload.get("all_scores", []),
            turn=payload.get("turn", 0),
            max_turns=payload.get("max_turns", 5),
            attack_intensity=payload.get("attack_intensity", 0.2),
            trust_erosion=payload.get("trust_erosion", 0.0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )
