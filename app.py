"""
FastAPI app exposing the OpenEnv HTTP API.
Endpoints: POST /reset, POST /step, GET /state, GET /health
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env import EmailTriageEnv, Action, Label

app = FastAPI(title="Email Triage OpenEnv", version="1.0.0")

# One environment instance per task (simple stateful server)
_envs: dict[str, EmailTriageEnv] = {}


def _get_env(task: str) -> EmailTriageEnv:
    if task not in _envs:
        _envs[task] = EmailTriageEnv(task_name=task)
    return _envs[task]


class ResetRequest(BaseModel):
    task: str = "medium"

class StepRequest(BaseModel):
    task:     str = "medium"
    email_id: int
    label:    str
    reply:    Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest):
    env = _get_env(req.task)
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    env = _get_env(req.task)
    try:
        action = Action(email_id=req.email_id, label=Label(req.label), reply=req.reply)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }


@app.get("/state")
def state(task: str = "medium"):
    env = _get_env(task)
    return env.state()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
