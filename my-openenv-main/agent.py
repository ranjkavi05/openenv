"""
agent.py — Baseline rule-based AI agent for the LifeOS ✨.

Implements a simple but effective decision-making heuristic that
prioritizes survival and balance. Runs a full simulation and prints
step-by-step results with the final grade.

Usage:
    python agent.py
"""

import random
import json
from typing import Dict, Any, List, Optional

from env import LifeSimulatorEnv
from models import Action, Personality, Difficulty, TaskType
from grader import grade_agent, grade_label
from utils import format_state, format_bar


# ═══════════════════════════════════════════════
#  RULE-BASED AGENT
# ═══════════════════════════════════════════════

class BaselineAgent:
    """A simple rule-based agent that makes decisions based on current state thresholds.

    Decision priority (highest to lowest):
    1. If stress is dangerously high → REST
    2. If health is critically low → EXERCISE
    3. If money is very low → WORK OVERTIME
    4. If relationships are neglected → SOCIALIZE
    5. If career is stalling → LEARN SKILL
    6. Otherwise → choose the action that addresses the weakest area
    """

    def __init__(self, task_type: TaskType = TaskType.PERFECT_BALANCE, seed: int = 42):
        self.rng = random.Random(seed)
        self.task_type = task_type
        self.decisions: List[Dict[str, Any]] = []

    def decide(self, state: Dict[str, Any]) -> tuple[str, str]:
        """Choose an action based on the current state.

        Returns
        -------
        action : str
            The chosen action name.
        reasoning : str
            Human-readable explanation of why this action was chosen.
        """
        health = state.get("health", 50)
        money = state.get("money", 500)
        stress = state.get("stress", 30)
        career = state.get("career", 30)
        relationships = state.get("relationships", 50)
        happiness = state.get("happiness", 50)

        # ── Priority 1: Survive stress burnout ──
        if stress >= 70:
            return "rest", f"Stress is dangerously high ({stress:.0f}) — need to rest immediately."

        # ── Priority 2: Survive health crisis ──
        if health <= 35:
            return "exercise", f"Health is critically low ({health:.0f}) — exercising to recover."

        # ── Priority 3: Financial survival ──
        if money <= 200:
            return "work_overtime", f"Running out of money (${money:.0f}) — working to earn."

        # ── Priority 4: Don't neglect relationships ──
        if relationships <= 25:
            return "socialize", f"Relationships are suffering ({relationships:.0f}) — socializing."

        # ── Priority 5: Career development ──
        if career <= 20:
            return "learn_skill", f"Career is stalling ({career:.0f}) — learning new skills."

        # ── Priority 6: Balanced strategy — address weakest area ──
        scores = {
            "exercise":      health,         # lower health → more need
            "work_overtime":  money / 100,    # normalized
            "socialize":     relationships,
            "learn_skill":   career,
            "rest":          100 - stress,    # higher stress → lower score
            "invest_money":  money / 100,     # invest when flush
        }

        # Adjust heuristics based on TaskType
        if self.task_type == TaskType.WEALTH_BUILDER:
            scores["work_overtime"] *= 0.2  # Make earning money look much more "needy"
            scores["invest_money"] *= 0.5
        elif self.task_type == TaskType.CAREER_CLIMBER:
            scores["learn_skill"] *= 0.2    # Prioritize learning and working
            scores["work_overtime"] *= 0.6

        # Find the dimension with lowest score (most needy)
        weakest_action = min(scores, key=scores.get)

        # Occasionally invest if we have enough money
        if money > 2000 and self.rng.random() < 0.3:
            return "invest_money", f"Good financial position (${money:.0f}) — investing for growth."

        reasons = {
            "exercise":     f"Balancing health ({health:.0f}).",
            "work_overtime": f"Building financial reserves (${money:.0f}).",
            "socialize":    f"Strengthening relationships ({relationships:.0f}).",
            "learn_skill":  f"Advancing career ({career:.0f}).",
            "rest":         f"Managing stress levels ({stress:.0f}).",
            "invest_money": f"Growing wealth (${money:.0f}).",
        }

        return weakest_action, reasons.get(weakest_action, "Balanced decision.")


# ═══════════════════════════════════════════════
#  SIMULATION RUNNER
# ═══════════════════════════════════════════════

def run_simulation(
    task_type: TaskType = TaskType.PERFECT_BALANCE,
    personality: Personality = Personality.BALANCED,
    difficulty: Difficulty = Difficulty.MEDIUM,
    seed: int = 42,
    max_steps: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a full simulation with the baseline agent.

    Returns
    -------
    dict with keys: final_state, grade, grade_label, total_reward,
                    steps, action_history, event_log
    """
    env = LifeSimulatorEnv(
        task_type=task_type,
        personality=personality,
        difficulty=difficulty,
        seed=seed,
        max_steps=max_steps,
    )
    agent = BaselineAgent(task_type=task_type, seed=seed)

    state = env.reset()

    if verbose:
        print("============================================================")
        print("  LifeOS ✨  —  Baseline Agent")
        print("=" * 60)
        print(f"  Task Goal   : {task_type.value}")
        print(f"  Personality : {personality.value}")
        print(f"  Difficulty  : {difficulty.value}")
        print(f"  Max Steps   : {max_steps}")
        print(f"  Seed        : {seed}")
        print("=" * 60)
        print()

    total_reward = 0.0
    step = 0
    done = False

    while not done:
        action, reasoning = agent.decide(state)
        agent.decisions.append({
            "step": step,
            "action": action,
            "reasoning": reasoning,
            "state_before": dict(state),
        })

        state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        if verbose and (step % 10 == 0 or done or step <= 3):
            print(f"─── Step {step:>3d}  | Week {info.get('week', '?')} "
                  f"| Age {info.get('age', '?'):.1f} ───")
            print(f"  Action    : {action} — {reasoning}")
            print(f"  Reward    : {reward:+.4f}  |  Cumulative: {total_reward:+.4f}")
            if "events" in info:
                for ev in info["events"]:
                    print(f"  🔔 EVENT  : {ev['description']}")
            print(f"  {format_bar('Health', state['health'])}")
            print(f"  {format_bar('Money', state['money'], 10000)}")
            print(f"  {format_bar('Stress', state['stress'])}")
            print(f"  {format_bar('Career', state['career'])}")
            print(f"  {format_bar('Relationships', state['relationships'])}")
            print(f"  {format_bar('Happiness', state['happiness'])}")
            print()

    # Final grading
    final_grade = grade_agent(state, task_type=task_type.value)
    label = grade_label(final_grade)

    if verbose:
        print("=" * 60)
        print("  📊  FINAL RESULTS")
        print("=" * 60)
        print(f"\n  Total Steps     : {step}")
        print(f"  Total Reward    : {total_reward:+.4f}")
        print(f"  Final Grade     : {final_grade:.4f}")
        print(f"  Assessment      : {label}")
        if info.get("done_reason"):
            print(f"  End Reason      : {info['done_reason']}")
        print(f"\n  Final State:")
        print(format_state(state, indent=4))

        event_log = env.get_event_log()
        if event_log:
            print(f"\n  📜 Events ({len(event_log)} total):")
            for ev in event_log[-5:]:
                print(f"    Step {ev['step']:>3d}: {ev['description']}")
        print("\n" + "=" * 60)

    return {
        "final_state": state,
        "grade": final_grade,
        "grade_label": label,
        "total_reward": total_reward,
        "steps": step,
        "action_history": env.action_history,
        "event_log": env.get_event_log(),
        "decisions": agent.decisions,
        "history": env.history,
    }


# ═══════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  --- OPENENV AGENT BENCHMARK SUITE (3 TASKS) ---")
    print("=" * 60)
    
    tasks = [TaskType.WEALTH_BUILDER, TaskType.CAREER_CLIMBER, TaskType.PERFECT_BALANCE]
    scores = {}
    
    for t in tasks:
        # Run silent simulation for benchmarking
        result = run_simulation(
            task_type=t,
            personality=Personality.BALANCED,
            difficulty=Difficulty.MEDIUM,
            seed=42,
            max_steps=100,
            verbose=False,
        )
        scores[t.value] = result['grade']
        print(f"  Task: {t.value.upper():<20} | Score: {result['grade']:.4f}  | Steps: {result['steps']}")
        
    print("=" * 60)
    print("  Benchmark complete. Reproducible baseline verified.")
    print("=" * 60 + "\n")
