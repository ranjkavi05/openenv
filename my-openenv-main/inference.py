"""
inference.py - Main inference script for the LifeOS ✨.

Required by OpenEnv hackathon submission rules.
Uses the OpenAI Client with API_BASE_URL, MODEL_NAME, and HF_TOKEN
environment variables for LLM-based agent decisions.

Usage:
    python inference.py
"""

import os
import json
import time
import sys
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from openai import OpenAI

from env import LifeSimulatorEnv
from models import Personality, Difficulty, TaskType, VALID_ACTIONS
from grader import grade_agent, grade_label

# ==================================================
#  ENVIRONMENT CONFIGURATION
# ==================================================

# Load .env file if present (local development)
load_dotenv()

# Required environment variables per hackathon rules
API_BASE_URL: str = os.environ.get("API_BASE_URL", "")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")


def _create_client() -> OpenAI:
    """Initialize the OpenAI client using the required environment variables."""
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )


# ==================================================
#  PROMPTS
# ==================================================

TASK_DESCRIPTIONS: Dict[str, str] = {
    "wealth_builder": (
        "Maximize money (> $50,000) before simulation end without dying. "
        "Focus on earning and investing while keeping health above zero."
    ),
    "career_climber": (
        "Reach Career > 90 while surviving 150 weeks. High stress causes "
        "burnout penalties. Focus on career growth while managing stress."
    ),
    "perfect_balance": (
        "Maintain > 80 across Health, Career, Wealth, and Relationships "
        "simultaneously. Balance all life dimensions evenly."
    ),
}

SYSTEM_PROMPT = (
    "You are an AI agent playing a life simulation game. "
    "Each turn you must choose exactly ONE action from the list below.\n\n"
    "Available actions: work_overtime, exercise, invest_money, "
    "learn_skill, socialize, rest\n\n"
    "Action effects:\n"
    "- work_overtime: +money(+120), +career(+4), +stress(+12), "
    "  -health(-3), -happiness(-4), -relationships(-2)\n"
    "- exercise: +health(+8), -stress(-8), +happiness(+5), -money(-10)\n"
    "- invest_money: variable money returns (delayed 3-5 steps), "
    "  +stress(+4), costs $150 upfront\n"
    "- learn_skill: +career(+7), +happiness(+3), +stress(+5), "
    "  -money(-30), -health(-1)\n"
    "- socialize: +relationships(+10), +happiness(+8), -stress(-5), "
    "  -money(-40), +health(+1)\n"
    "- rest: +health(+5), -stress(-12), +happiness(+4), -career(-1)\n\n"
    "CRITICAL RULES:\n"
    "- If health reaches 0, you die (episode ends).\n"
    "- If stress reaches 100, you burn out (episode ends).\n"
    "- Respond with ONLY the action name. No explanation, no punctuation."
)


def _build_user_prompt(
    state: Dict[str, Any],
    task_type: str,
    step: int,
    max_steps: int,
) -> str:
    """Build the user prompt with current state and objective."""
    task_desc = TASK_DESCRIPTIONS.get(task_type, "Balance all life dimensions.")

    return (
        f"OBJECTIVE: {task_desc}\n\n"
        f"Current State (Step {step}/{max_steps}):\n"
        f"- Health: {state['health']:.1f}/100\n"
        f"- Money: ${state['money']:.0f}\n"
        f"- Stress: {state['stress']:.1f}/100\n"
        f"- Career: {state['career']:.1f}/100\n"
        f"- Relationships: {state['relationships']:.1f}/100\n"
        f"- Happiness: {state['happiness']:.1f}/100\n"
        f"- Age: {state['age']:.1f}\n\n"
        f"Choose your action:"
    )


# ==================================================
#  LLM-POWERED AGENT
# ==================================================

class LLMAgent:
    """Agent that uses an OpenAI-compatible LLM for decision making.

    Falls back to a deterministic rule-based strategy if the LLM
    is unreachable or returns an invalid action.
    """

    def __init__(self, client: OpenAI, task_type: TaskType):
        self.client = client
        self.task_type = task_type
        self.decisions: List[Dict[str, Any]] = []
        self._llm_available = bool(API_BASE_URL and MODEL_NAME and HF_TOKEN)

    # -- public API --

    def decide(
        self,
        state: Dict[str, Any],
        step: int,
        max_steps: int,
    ) -> Tuple[str, str]:
        """Choose an action based on the current state.

        Returns (action_name, reasoning_string).
        """
        if self._llm_available:
            action, reasoning = self._llm_decide(state, step, max_steps)
        else:
            action, reasoning = self._fallback_decide(state)
            reasoning = f"[Rule-based] {reasoning}"
        return action, reasoning

    # -- LLM decision --

    def _llm_decide(
        self,
        state: Dict[str, Any],
        step: int,
        max_steps: int,
    ) -> Tuple[str, str]:
        """Call the LLM to choose an action."""
        user_prompt = _build_user_prompt(
            state, self.task_type.value, step, max_steps
        )

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=20,
                temperature=0.3,
            )

            raw = response.choices[0].message.content.strip().lower()
            action = raw.strip().replace(".", "").replace(",", "").strip()

            # Validate action
            if action in VALID_ACTIONS:
                return action, f"LLM chose: {action}"

            # Try to find a valid action substring in the response
            for valid in VALID_ACTIONS:
                if valid in raw:
                    return valid, f"LLM chose: {valid} (parsed from '{raw[:30]}')"

            # LLM returned garbage -- fall back
            action, reason = self._fallback_decide(state)
            return action, f"[Fallback - invalid LLM output: '{raw[:30]}'] {reason}"

        except Exception as exc:
            action, reason = self._fallback_decide(state)
            return action, f"[Fallback - {str(exc)[:50]}] {reason}"

    # -- Rule-based fallback --

    def _fallback_decide(self, state: Dict[str, Any]) -> Tuple[str, str]:
        """Deterministic rule-based fallback when LLM is unavailable."""
        health = state.get("health", 50)
        money = state.get("money", 500)
        stress = state.get("stress", 30)
        career = state.get("career", 30)
        relationships = state.get("relationships", 50)

        # Priority 1 - survive stress burnout
        if stress >= 70:
            return "rest", f"Stress critical ({stress:.0f})"
        # Priority 2 - survive health crisis
        if health <= 35:
            return "exercise", f"Health critical ({health:.0f})"
        # Priority 3 - financial survival
        if money <= 200:
            return "work_overtime", f"Money critical (${money:.0f})"
        # Priority 4 - don't neglect relationships
        if relationships <= 25:
            return "socialize", f"Relationships critical ({relationships:.0f})"
        # Priority 5 - career stalling
        if career <= 20:
            return "learn_skill", f"Career critical ({career:.0f})"

        # Task-specific heuristics
        if self.task_type == TaskType.WEALTH_BUILDER:
            if money > 2000:
                return "invest_money", "Investing for wealth growth"
            return "work_overtime", "Working to build wealth"

        if self.task_type == TaskType.CAREER_CLIMBER:
            if stress > 50:
                return "rest", "Managing stress for career sustainability"
            return "learn_skill", "Learning to climb career"

        # Perfect balance - address weakest dimension
        scores = {
            "exercise": health,
            "work_overtime": money / 100,
            "socialize": relationships,
            "learn_skill": career,
            "rest": 100 - stress,
        }
        weakest = min(scores, key=scores.get)  # type: ignore[arg-type]
        return weakest, f"Balancing weakest area ({weakest})"


# ==================================================
#  INFERENCE RUNNER
# ==================================================

def run_task(
    client: OpenAI,
    task_type: TaskType,
    max_steps: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a single task end-to-end and return results including the grade."""
    env = LifeSimulatorEnv(
        task_type=task_type,
        personality=Personality.BALANCED,
        difficulty=Difficulty.MEDIUM,
        seed=seed,
        max_steps=max_steps,
    )
    agent = LLMAgent(client=client, task_type=task_type)

    state = env.reset()
    total_reward = 0.0
    step = 0
    done = False

    print(f"\n  Running task: {task_type.value}")
    print(f"  {'-' * 50}")

    while not done:
        action, reasoning = agent.decide(state, step, max_steps)
        agent.decisions.append({
            "step": step,
            "action": action,
            "reasoning": reasoning,
        })

        state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        # Print progress every 25 steps and at the end
        if step % 25 == 0 or done:
            print(
                f"    Step {step:>3d} | Action: {action:<15s} "
                f"| Reward: {reward:+.4f} | Cumulative: {total_reward:+.4f}"
            )

    # Grade the final state
    final_grade = grade_agent(state, task_type=task_type.value)
    label = grade_label(final_grade)

    print(f"  -- Result: Grade = {final_grade:.4f}  |  {label}")
    print(f"  -- Steps: {step}  |  Total Reward: {total_reward:+.4f}")

    return {
        "task": task_type.value,
        "final_state": state,
        "grade": final_grade,
        "grade_label": label,
        "total_reward": round(total_reward, 4),
        "steps": step,
        "decisions": agent.decisions,
    }


# ==================================================
#  MAIN ENTRY POINT
# ==================================================

def main() -> None:
    """Main inference entry point - runs all 3 required tasks."""
    start_time = time.time()

    print("============================================================")
    print("  LifeOS ✨ - Inference Script")
    print("=" * 60)
    print(
        f"  API_BASE_URL : {API_BASE_URL[:40]}..."
        if len(API_BASE_URL) > 40
        else f"  API_BASE_URL : {API_BASE_URL or '(not set)'}"
    )
    print(f"  MODEL_NAME   : {MODEL_NAME or '(not set)'}")
    print(
        f"  HF_TOKEN     : {'*' * 8}...{HF_TOKEN[-4:]}"
        if len(HF_TOKEN) > 4
        else f"  HF_TOKEN     : {'(set)' if HF_TOKEN else '(not set)'}"
    )
    print("=" * 60)

    # Warn if env vars are missing
    if not all([API_BASE_URL, MODEL_NAME, HF_TOKEN]):
        print(
            "\n  WARNING: One or more required env vars not set.\n"
            "     Will use deterministic rule-based fallback agent.\n"
            "     Set API_BASE_URL, MODEL_NAME, HF_TOKEN for LLM mode.\n"
        )

    # Create OpenAI client
    client = _create_client()

    # -- Run all 3 tasks --
    tasks = [
        TaskType.WEALTH_BUILDER,
        TaskType.CAREER_CLIMBER,
        TaskType.PERFECT_BALANCE,
    ]
    results: List[Dict[str, Any]] = []

    for task in tasks:
        result = run_task(client, task, max_steps=100, seed=42)
        results.append(result)

    # -- Summary --
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)

    all_valid = True
    for r in results:
        grade = r["grade"]
        valid = 0.0 <= grade <= 1.0
        if not valid:
            all_valid = False
        status = "[PASS]" if valid else "[FAIL]"
        print(
            f"  {status} {r['task']:<20s} "
            f"| Grade: {grade:.4f} "
            f"| Steps: {r['steps']}"
        )

    print(f"\n  Total time: {elapsed:.1f}s")
    print(
        f"  All scores in 0.0-1.0 range: "
        f"{'YES' if all_valid else 'NO'}"
    )
    print("=" * 60)

    # JSON output for automated parsing
    output = {
        "tasks": {
            r["task"]: {"grade": r["grade"], "steps": r["steps"]}
            for r in results
        },
        "elapsed_seconds": round(elapsed, 2),
        "all_valid": all_valid,
    }

    print(f"\n{json.dumps(output, indent=2)}")

    return


if __name__ == "__main__":
    main()
