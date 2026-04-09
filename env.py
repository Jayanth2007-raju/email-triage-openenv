"""
Email Triage OpenEnv Environment
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Label(str, Enum):
    URGENT = "urgent"
    FOLLOW_UP = "follow_up"
    SPAM = "spam"
    ARCHIVE = "archive"
    REPLY = "reply"


class Action(BaseModel):
    email_id: int
    label: Label
    reply: Optional[str] = None


class Observation(BaseModel):
    emails: list[dict]
    step_number: int
    task_hint: str


class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    reason: str


EMAILS = [
    {"id": 1, "from": "ceo@company.com",
     "subject": "URGENT: Board meeting TODAY 3pm",
     "body": "Confirm attendance.",
     "true_label": Label.URGENT, "reply_keywords": ["confirm", "attend", "3pm"]},

    {"id": 2, "from": "newsletter",
     "subject": "Weekly digest",
     "body": "Click here",
     "true_label": Label.SPAM, "reply_keywords": []},

    {"id": 3, "from": "client",
     "subject": "Need proposal",
     "body": "Send by Friday",
     "true_label": Label.REPLY, "reply_keywords": ["proposal", "send", "friday"]},
]


TASKS = {
    "easy": {"email_ids": [1, 2, 3], "max_steps": 3, "hint": "Classify emails"},
    "medium": {"email_ids": [1, 2, 3], "max_steps": 3, "hint": "Classify emails"},
    "hard": {"email_ids": [1, 2, 3], "max_steps": 3, "hint": "Classify + reply"},
}


class EmailTriageEnv:

    def __init__(self, task_name="medium"):
        self._task = TASKS[task_name]
        self._emails = {e["id"]: e for e in EMAILS}
        self._actions = {}
        self._step = 0
        self._done = False

    def reset(self):
        self._actions = {}
        self._step = 0
        self._done = False
        return self._obs()

    def step(self, action: Action):

        if self._done:
            raise RuntimeError("Episode done")

        email = self._emails.get(action.email_id)

        if not email:
            return self._obs(), Reward(value=0.01, reason="invalid"), False, {}

        self._actions[action.email_id] = action
        self._step += 1

        reward = self._reward(email, action)

        remaining = set(self._task["email_ids"]) - set(self._actions)
        self._done = not remaining or self._step >= self._task["max_steps"]

        return self._obs(), reward, self._done, {}

    def state(self):
        return {"step": self._step, "done": self._done}

    def _reward(self, email, action):

        correct = action.label == email["true_label"]

        # FIXED LABEL SCORE
        if correct:
            label_score = 0.99
        else:
            label_score = 0.01

        reply_score = 0.01

        if email["true_label"] in (Label.REPLY, Label.URGENT):
            if action.reply:
                reply_score = 0.5

        total = round(0.6 * label_score + 0.4 * reply_score, 2)

        # FINAL CLAMP
        if total <= 0.0:
            total = 0.01
        elif total >= 1.0:
            total = 0.99

        return Reward(value=total, reason="computed")

    def _obs(self):
        return Observation(
            emails=[{k: v for k, v in e.items() if k != "true_label"} for e in self._emails.values()],
            step_number=self._step,
            task_hint=self._task["hint"],
        )
