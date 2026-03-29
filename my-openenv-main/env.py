"""
env.py — Core Environment for the LifeOS ✨.

Implements an OpenEnv-compatible reinforcement-learning environment that
simulates real-world human life decisions. Integrates the reward system,
time progression, personality modifiers, dynamic events, and difficulty
levels into one cohesive engine.

OpenEnv interface:
    reset()        → initial state dict
    step(action)   → (state, reward, done, info)
    state()        → current state dict
"""

from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Action, Difficulty, LifeState, Personality, StepResult, VALID_ACTIONS, TaskType
)
from utils import clamp, normalize, imbalance_penalty, weighted_average
from personalities import get_profile, get_action_multiplier, PersonalityProfile
from events import EventSystem, EventRecord


# ═══════════════════════════════════════════════
#  REWARD WEIGHTS & CONSTANTS
# ═══════════════════════════════════════════════

# These weights define how much each life dimension contributes to the
# per-step reward. The sum is 1.0 to keep the reward interpretable.
REWARD_WEIGHTS = {
    "health":        0.25,    # physical well-being is paramount
    "career":        0.20,    # professional growth matters
    "relationships": 0.20,    # social bonds are essential
    "money":         0.15,    # financial stability (but not greed)
    "stress_inv":    0.20,    # inverse stress — lower stress = higher reward
}

# Consistency bonus: if the agent maintains balanced stats for N consecutive
# steps, it earns a small bonus each step to encourage sustained balance.
CONSISTENCY_WINDOW = 5         # number of recent steps to evaluate
CONSISTENCY_THRESHOLD = 0.15   # max allowed imbalance to still earn bonus
CONSISTENCY_BONUS = 0.05       # reward bonus per balanced step

# Burnout penalty: if stress stays above this threshold for N consecutive
# steps, apply an escalating penalty to discourage chronic overwork.
BURNOUT_STRESS_THRESHOLD = 70.0
BURNOUT_WINDOW = 3
BURNOUT_PENALTY_PER_STEP = 0.08


# ═══════════════════════════════════════════════
#  DIFFICULTY CONFIGURATION
# ═══════════════════════════════════════════════

DIFFICULTY_CONFIG = {
    Difficulty.EASY: {
        "stress_gain_mult":  0.7,     # gentler stress accumulation
        "health_decay":      0.05,    # slow natural health decay
        "money_loss_mult":   0.5,     # softer financial penalties
        "reward_bonus":      0.02,    # slight reward bump
    },
    Difficulty.MEDIUM: {
        "stress_gain_mult":  1.0,
        "health_decay":      0.10,
        "money_loss_mult":   1.0,
        "reward_bonus":      0.0,
    },
    Difficulty.HARD: {
        "stress_gain_mult":  1.4,     # stress piles on fast
        "health_decay":      0.20,    # health degrades quicker
        "money_loss_mult":   1.3,     # harsher financial hits
        "reward_bonus":      -0.02,   # slight reward reduction
    },
}


# ═══════════════════════════════════════════════
#  TIME PROGRESSION CONSTANTS
# ═══════════════════════════════════════════════

WEEKS_PER_YEAR = 52
AGE_PER_STEP = 1.0 / WEEKS_PER_YEAR       # each step ≈ 1 week


# ═══════════════════════════════════════════════
#  ACTION EFFECTS TABLE
# ═══════════════════════════════════════════════

def _base_action_effects() -> Dict[str, Dict[str, float]]:
    """Return the base effect deltas for each action.

    Each action affects multiple variables to create an interconnected
    system where trade-offs are necessary.
    """
    return {
        "work_overtime": {
            "money":         +120,
            "career":        +4,
            "stress":        +12,
            "health":        -3,
            "happiness":     -4,
            "relationships": -2,
        },
        "exercise": {
            "health":        +8,
            "stress":        -8,
            "happiness":     +5,
            "money":         -10,       # gym costs / time cost
            "career":        +0,
            "relationships": +1,
        },
        "invest_money": {
            # Base effect — actual returns are randomized in _apply_invest()
            "money":         0,         # computed dynamically
            "stress":        +4,
            "career":        +1,
            "happiness":     +0,
            "health":        +0,
            "relationships": -1,
        },
        "learn_skill": {
            "career":        +7,
            "happiness":     +3,
            "stress":        +5,
            "money":         -30,       # course / book costs
            "health":        -1,
            "relationships": -1,
        },
        "socialize": {
            "relationships": +10,
            "happiness":     +8,
            "stress":        -5,
            "money":         -40,       # going out costs money
            "career":        +1,        # networking
            "health":        +1,
        },
        "rest": {
            "health":        +5,
            "stress":        -12,
            "happiness":     +4,
            "money":         -5,        # not earning
            "career":        -1,        # stalling
            "relationships": +2,        # quality time
        },
    }


# ═══════════════════════════════════════════════
#  MAIN ENVIRONMENT CLASS
# ═══════════════════════════════════════════════

class LifeSimulatorEnv:
    """OpenEnv-compatible life simulation environment.

    Parameters
    ----------
    personality : Personality
        Modifies action impacts, rewards, and risk profiles.
    difficulty : Difficulty
        Controls randomness, event frequency, and trade-off severity.
    seed : int or None
        Random seed for reproducibility.
    max_steps : int
        Maximum steps before forced termination (prevents infinite runs).
    """

    def __init__(
        self,
        task_type: TaskType = TaskType.PERFECT_BALANCE,
        personality: Personality = Personality.BALANCED,
        difficulty: Difficulty = Difficulty.MEDIUM,
        seed: Optional[int] = None,
        max_steps: int = 200,
    ):
        self.task_type = task_type
        self.personality = personality
        self.difficulty = difficulty
        self.max_steps = max_steps

        # Reproducible RNG
        self.rng = random.Random(seed)
        self._seed = seed

        # Sub-systems
        self.profile: PersonalityProfile = get_profile(personality)
        self.events = EventSystem(difficulty=difficulty, rng=self.rng)
        self.diff_cfg = DIFFICULTY_CONFIG[difficulty]

        # State
        self._state: LifeState = LifeState()
        self._done: bool = False

        # History (for reward consistency tracking and UI playback)
        self._history: List[Dict[str, Any]] = []
        self._action_history: List[str] = []
        self._reward_history: List[float] = []
        self._high_stress_streak: int = 0  # consecutive high-stress steps

        # Delayed investment returns queue: list of (step_to_pay, amount)
        self._pending_investments: List[Tuple[int, float]] = []

    # ─────────────────────────────────────────
    #  OpenEnv Interface
    # ─────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        """Reset the environment to its initial state.

        Returns the initial observation as a dictionary.
        """
        self.rng = random.Random(self._seed)
        self._state = LifeState()
        self._done = False
        self._history.clear()
        self._action_history.clear()
        self._reward_history.clear()
        self._high_stress_streak = 0
        self._pending_investments.clear()
        self.events = EventSystem(difficulty=self.difficulty, rng=self.rng)
        self._history.append(self._state.to_dict())
        return self.state()

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Advance the simulation by one step (≈ 1 week).

        Parameters
        ----------
        action : str
            One of the valid action names (see models.VALID_ACTIONS).

        Returns
        -------
        state : dict   — current observation
        reward : float — computed reward for this step
        done : bool    — whether a terminal condition is met
        info : dict    — metadata (events, reasoning, grade, etc.)
        """
        if self._done:
            return self.state(), 0.0, True, {"error": "Episode already finished."}

        if action not in VALID_ACTIONS:
            return self.state(), -1.0, False, {"error": f"Invalid action: {action}"}

        info: Dict[str, Any] = {"action": action, "step": self._state.step_count}

        # 1) Apply action effects (personality-modified)
        self._apply_action(action)

        # 2) Time progression
        self._tick_time()

        # 3) Process delayed investment returns
        self._process_investments()

        # 4) Random events
        triggered = self.events.check_events(self._state)
        if triggered:
            self.events.apply_events(self._state, triggered)
            info["events"] = [
                {"event": e.event_type.value, "description": e.description,
                 "effects": e.effects}
                for e in triggered
            ]

        # 5) Clamp all state values to valid ranges
        self._clamp_state()

        # 6) Check termination
        self._done = self._check_done()
        info["done_reason"] = self._done_reason() if self._done else None

        # 7) Compute reward
        reward = self._compute_reward()
        reward += self.diff_cfg["reward_bonus"]
        reward += self.profile.reward_bias

        # 8) Record history
        self._history.append(self._state.to_dict())
        self._action_history.append(action)
        self._reward_history.append(reward)

        info["cumulative_reward"] = sum(self._reward_history)
        info["week"] = self._state.week
        info["age"] = self._state.age

        return self.state(), round(reward, 4), self._done, info

    def state(self) -> Dict[str, Any]:
        """Return the current observation as a plain dictionary (filtered for OpenEnv compliance)."""
        full_state = self._state.to_dict()
        # Return only the keys defined in the openenv.yaml observation space
        return {
            k: v for k, v in full_state.items() 
            if k in ["age", "health", "money", "stress", "career", "relationships", "happiness"]
        }

    # ─────────────────────────────────────────
    #  Action Application
    # ─────────────────────────────────────────

    def _apply_action(self, action: str) -> None:
        """Apply the chosen action's effects to the current state.

        Effects are scaled by the active personality profile.
        """
        effects = _base_action_effects()[action]
        mult = get_action_multiplier(self.profile, action)
        stress_mult = self.diff_cfg["stress_gain_mult"]

        for attr, delta in effects.items():
            if attr == "money" and action == "invest_money":
                # Investment is handled separately for risk-reward logic
                continue
            scaled = delta * mult
            # Extra difficulty scaling for stress gains
            if attr == "stress" and delta > 0:
                scaled *= stress_mult
            current = getattr(self._state, attr)
            setattr(self._state, attr, current + scaled)

        # Handle investment separately
        if action == "invest_money":
            self._apply_invest()

        # Work output scaling for money from work_overtime
        if action == "work_overtime":
            extra = 120 * (self.profile.work_output - 1.0)
            self._state.money += extra

        # Career growth modifier
        if action in ("learn_skill", "work_overtime"):
            career_extra = 2.0 * (self.profile.career_growth - 1.0)
            self._state.career += career_extra

        # Social effectiveness
        if action == "socialize":
            rel_extra = 5.0 * (self.profile.social_effectiveness - 1.0)
            self._state.relationships += rel_extra

    def _apply_invest(self) -> None:
        """Handle investment with risk-reward logic.

        - Base cost: -150 money upfront
        - Returns arrive 3–5 steps later with randomized outcome
        - Investment risk scaled by personality
        """
        cost = 150
        if self._state.money < cost:
            # Can't afford to invest — minor stress from frustration
            self._state.stress += 3
            return

        self._state.money -= cost
        risk = self.profile.investment_risk

        # Determine return multiplier with risk-scaled variance
        # Mean return is 1.3× (30% gain), but variance increases with risk
        mean_return = 1.3
        variance = 0.5 * risk       # higher risk → wider swings
        multiplier = self.rng.gauss(mean_return, variance)
        multiplier = max(0.0, multiplier)   # can't lose more than invested

        payout = cost * multiplier
        delay = self.rng.randint(3, 5)     # delayed returns
        pay_step = self._state.step_count + delay
        self._pending_investments.append((pay_step, payout))

    def _process_investments(self) -> None:
        """Pay out any matured investments."""
        remaining = []
        for pay_step, amount in self._pending_investments:
            if self._state.step_count >= pay_step:
                self._state.money += amount
                self._state.happiness += 3 if amount > 150 else -2
            else:
                remaining.append((pay_step, amount))
        self._pending_investments = remaining

    # ─────────────────────────────────────────
    #  Time Progression
    # ─────────────────────────────────────────

    def _tick_time(self) -> None:
        """Advance time by one step (≈ 1 week).

        Applies natural aging effects:
        - Age increases
        - Health decays slightly if not actively exercised
        - Stress personality modifier applied
        - Career grows a tiny amount passively
        """
        self._state.step_count += 1
        self._state.week += 1
        self._state.age += AGE_PER_STEP

        # Natural health decay (aging)
        decay = self.diff_cfg["health_decay"]
        # Decay accelerates slightly with age
        age_factor = 1.0 + max(0, (self._state.age - 40)) * 0.005
        self._state.health -= decay * age_factor

        # Personality baseline stress modifier
        self._state.stress += self.profile.stress_modifier * 0.3

        # Passive career drift (tiny growth from existing experience)
        self._state.career += 0.1

        # Long-term stress damage: if stress stays high, health erodes
        if self._state.stress > BURNOUT_STRESS_THRESHOLD:
            self._high_stress_streak += 1
            if self._high_stress_streak >= BURNOUT_WINDOW:
                dmg = 2.0 * (self._high_stress_streak - BURNOUT_WINDOW + 1)
                self._state.health -= dmg
        else:
            self._high_stress_streak = max(0, self._high_stress_streak - 1)

        # Happiness natural regression toward 50
        self._state.happiness += (50 - self._state.happiness) * 0.02

    # ─────────────────────────────────────────
    #  Reward Computation
    # ─────────────────────────────────────────

    def _compute_reward(self) -> float:
        """Compute the step reward based on the active TaskType objective."""
        s = self._state

        # Check for death penalty
        if s.health <= 0:
            return -1.0

        if self.task_type == TaskType.WEALTH_BUILDER:
            # Dense reward: Pure wealth accumulation delta
            prev_money = self._history[-1]["money"] if self._history else 1000.0
            money_delta = s.money - prev_money
            # Dense partial progress: +0.01 per $100 formed
            return clamp(money_delta / 10000.0, -1.0, 1.0)
            
        elif self.task_type == TaskType.CAREER_CLIMBER:
            # Dense reward: Career growth delta minus stress risks
            prev_career = self._history[-1]["career"] if self._history else 30.0
            career_delta = s.career - prev_career
            r = clamp(career_delta / 10.0, -1.0, 1.0)
            if s.stress > 80:
                r -= 0.5  # Substantial penalty for flirting with burnout
            return r

        # Defaults to TaskType.PERFECT_BALANCE
        # ── 1. Weighted score ──
        norm_vals = {
            "health":    normalize(s.health, 0, 100),
            "career":    normalize(s.career, 0, 100),
            "relationships": normalize(s.relationships, 0, 100),
            "money":     normalize(s.money, 0, 10000),
            "stress_inv": 1.0 - normalize(s.stress, 0, 100),
        }
        weighted = weighted_average(norm_vals, REWARD_WEIGHTS)

        # ── 2. Imbalance penalty ──
        core_values = [
            norm_vals["health"],
            norm_vals["career"],
            norm_vals["relationships"],
            normalize(s.happiness, 0, 100),
        ]
        imbalance = imbalance_penalty(core_values)
        imbalance_cost = imbalance * 0.15       # scale penalty magnitude

        # ── 3. Consistency bonus ──
        consistency = 0.0
        if len(self._reward_history) >= CONSISTENCY_WINDOW:
            recent_states = self._history[-CONSISTENCY_WINDOW:]
            recent_imbalances = []
            for h in recent_states:
                vals = [
                    normalize(h["health"], 0, 100),
                    normalize(h["career"], 0, 100),
                    normalize(h["relationships"], 0, 100),
                    normalize(h["happiness"], 0, 100),
                ]
                recent_imbalances.append(imbalance_penalty(vals))
            if all(ib < CONSISTENCY_THRESHOLD for ib in recent_imbalances):
                consistency = CONSISTENCY_BONUS

        # ── 4. Burnout penalty ──
        burnout = 0.0
        if self._high_stress_streak >= BURNOUT_WINDOW:
            burnout = BURNOUT_PENALTY_PER_STEP * (
                self._high_stress_streak - BURNOUT_WINDOW + 1
            )

        # ── Final reward ──
        reward = weighted - imbalance_cost + consistency - burnout
        return clamp(reward, -1.0, 1.0)

    # ─────────────────────────────────────────
    #  Termination Logic
    # ─────────────────────────────────────────

    def _check_done(self) -> bool:
        """Check if the episode should terminate."""
        if self._state.health <= 0:
            return True
        if self._state.stress >= 100:
            return True
        if self._state.step_count >= self.max_steps:
            return True
        return False

    def _done_reason(self) -> str:
        """Return a human-readable description of why the episode ended."""
        if self._state.health <= 0:
            return "Health reached zero — life ended."
        if self._state.stress >= 100:
            return "Stress overload — complete burnout."
        if self._state.step_count >= self.max_steps:
            return "Maximum simulation length reached."
        return "Unknown"

    # ─────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────

    def _clamp_state(self) -> None:
        """Clamp all state variables to their valid ranges."""
        s = self._state
        s.health = clamp(s.health, 0, 100)
        s.stress = clamp(s.stress, 0, 100)
        s.career = clamp(s.career, 0, 100)
        s.relationships = clamp(s.relationships, 0, 100)
        s.happiness = clamp(s.happiness, 0, 100)
        s.money = max(0, s.money)              # money can't go negative

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Full state history for timeline visualization."""
        return list(self._history)

    @property
    def action_history(self) -> List[str]:
        """Ordered list of actions taken."""
        return list(self._action_history)

    @property
    def reward_history(self) -> List[float]:
        """Ordered list of rewards received."""
        return list(self._reward_history)

    def get_event_log(self) -> List[Dict]:
        """Return the full event log."""
        return self.events.get_log()
