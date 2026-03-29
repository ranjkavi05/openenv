from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, Optional

from env import LifeSimulatorEnv
from models import TaskType, Personality, Difficulty, VALID_ACTIONS

app = FastAPI(title="OpenEnv LifeOS ✨ API")

# Global env instance for evaluation
env_instance: Optional[LifeSimulatorEnv] = None

@app.get("/")
def home():
    return {
        "status": "online", 
        "message": "LifeOS ✨ API is running! Access /docs for endpoint details. Ready for OpenEnv Grading."
    }

class ResetRequest(BaseModel):
    task_type: str = "perfect_balance"
    personality: str = "balanced"
    difficulty: str = "medium"
    seed: int = 42
    max_steps: int = 150
    model_config = ConfigDict(extra='allow')

class StepRequest(BaseModel):
    action: str
    model_config = ConfigDict(extra='allow')

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    global env_instance
    if req is None:
        req = ResetRequest()
        
    try:
        task = TaskType(req.task_type)
    except ValueError:
        task = TaskType.PERFECT_BALANCE
        
    try:
        pers = Personality(req.personality)
    except ValueError:
        pers = Personality.BALANCED
        
    try:
        diff = Difficulty(req.difficulty)
    except ValueError:
        diff = Difficulty.MEDIUM

    env_instance = LifeSimulatorEnv(
        task_type=task,
        personality=pers,
        difficulty=diff,
        seed=req.seed,
        max_steps=req.max_steps
    )
    obs = env_instance.reset()
    return {"observation": obs}

@app.post("/step")
def step(req: StepRequest):
    global env_instance
    if not env_instance:
        # Fallback initialization if step is called without reset
        reset(ResetRequest())
        
    if req.action not in VALID_ACTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid action: {req.action}. Valid actions: {VALID_ACTIONS}")
        
    obs, reward, done, info = env_instance.step(req.action)
    return {
        "observation": obs,
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }

@app.get("/state")
def state():
    if not env_instance:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return env_instance.state()
