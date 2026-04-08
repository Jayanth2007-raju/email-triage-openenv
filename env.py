"""
Email Triage OpenEnv Environment
Real-world task: triage a cluttered inbox — label, prioritize, and draft replies.
"""

from __future__ import annotations
import random
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Pydantic models ──────────────────────────────────────────────────────────

class Label(str, Enum):
    URGENT    = "urgent"
    FOLLOW_UP = "follow_up"
    SPAM      = "spam"
    ARCHIVE   = "archive"
    REPLY     = "reply"

class Action(BaseModel):
    email_id: int          = Field(..., description="ID of the email to act on")
    label:    Label        = Field(..., description="Label to assign")
    reply:    Optional[str]= Field(None, description="Draft reply text (required when label=reply)")

class Observation(BaseModel):
    emails:      list[dict]= Field(..., description="List of email dicts with id, from, subject, body")
    step_number: int       = Field(..., description="Current step in the episode")
    task_hint:   str       = Field(..., description="Natural language task instruction")

class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0, description="Reward for this step")
    reason: str  = Field(..., description="Explanation of the reward")


# ── Email fixtures ────────────────────────────────────────────────────────────

EMAILS = [
    {
        "id": 1, "from": "ceo@company.com",
        "subject": "URGENT: Board meeting rescheduled to TODAY 3pm",
        "body": "Hi team, the board meeting has been moved to today at 3 PM. Please confirm attendance immediately.",
        "true_label": Label.URGENT, "reply_keywords": ["confirm", "attend", "3pm"]
    },
    {
        "id": 2, "from": "noreply@newsletter.io",
        "subject": "Your weekly digest is ready!",
        "body": "Click here to read your weekly newsletter. Unsubscribe at any time.",
        "true_label": Label.SPAM, "reply_keywords": []
    },
    {
        "id": 3, "from": "client@bigcorp.com",
        "subject": "Re: Project proposal — need your response",
        "body": "We've been waiting for your updated proposal for 2 weeks. We need it by EOD Friday or we'll go elsewhere.",
        "true_label": Label.REPLY, "reply_keywords": ["proposal", "send", "friday", "deadline"]
    },
    {
        "id": 4, "from": "hr@company.com",
        "subject": "Annual performance review form due next month",
        "body": "Please complete your self-review form. Deadline is the 15th of next month.",
        "true_label": Label.FOLLOW_UP, "reply_keywords": []
    },
    {
        "id": 5, "from": "support@saas.com",
        "subject": "Your invoice #4821 is ready",
        "body": "Your invoice for $49/mo is attached. No action required.",
        "true_label": Label.ARCHIVE, "reply_keywords": []
    },
    {
        "id": 6, "from": "manager@company.com",
        "subject": "Server is DOWN — production incident",
        "body": "Our main API server is throwing 500s. Customers are impacted. Please jump in immediately.",
        "true_label": Label.URGENT, "reply_keywords": ["on it", "investigating", "fix", "incident"]
    },
    {
        "id": 7, "from": "sales@spam-corp.net",
        "subject": "Make $5000/day working from home!!",
        "body": "LIMITED OFFER: Click now to start earning big money! Don't miss out!",
        "true_label": Label.SPAM, "reply_keywords": []
    },
    {
        "id": 8, "from": "colleague@company.com",
        "subject": "Can you review my PR when you get a chance?",
        "body": "Hi, I've opened a PR on the auth module. No rush, but would love your feedback this week.",
        "true_label": Label.FOLLOW_UP, "reply_keywords": []
    },
]


# ── Task definitions ──────────────────────────────────────────────────────────

TASKS = {
    "easy": {
        "name": "easy",
        "description": "Label 3 clearly distinct emails (spam vs urgent vs archive).",
        "email_ids": [2, 6, 5],  # spam, urgent, archive
        "max_steps": 3,
        "hint": "Label each email correctly: spam, urgent, or archive.",
    },
    "medium": {
        "name": "medium",
        "description": "Label 5 emails including ambiguous ones (follow_up vs reply vs urgent).",
        "email_ids": [1, 2, 3, 4, 5],
        "max_steps": 5,
        "hint": "Triage all 5 emails. Some need replies, some are urgent, some can be archived or followed up.",
    },
    "hard": {
        "name": "hard",
        "description": "Label all 8 emails AND write draft replies for emails labeled 'reply' or 'urgent'.",
        "email_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "max_steps": 8,
        "hint": (
            "Triage all 8 emails. For every email you label 'reply' or 'urgent', "
            "you MUST also write a short draft reply covering the key points."
        ),
    },
}


# ── Environment ───────────────────────────────────────────────────────────────

class EmailTriageEnv:
    """OpenEnv-compatible email triage environment."""

    def __init__(self, task_name: str = "medium"):
        assert task_name in TASKS, f"task_name must be one of {list(TASKS)}"
        self._task_cfg   = TASKS[task_name]
        self._task_name  = task_name
        self._email_pool = {e["id"]: e for e in EMAILS}
        self._actions_taken: dict[int, Action] = {}
        self._step       = 0
        self._done       = False

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        self._actions_taken = {}
        self._step          = 0
        self._done          = False
        return self._build_obs()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        email = self._email_pool.get(action.email_id)
        if email is None:
            reward = Reward(value=0.0, reason=f"Unknown email_id {action.email_id}")
            obs    = self._build_obs()
            return obs, reward, False, {"error": f"Unknown email_id {action.email_id}"}

        if action.email_id in self._actions_taken:
            reward = Reward(value=0.0, reason=f"Email {action.email_id} already acted on")
            obs    = self._build_obs()
            return obs, reward, False, {"warning": "duplicate action"}

        self._actions_taken[action.email_id] = action
        self._step += 1

        reward = self._compute_reward(email, action)

        remaining = set(self._task_cfg["email_ids"]) - set(self._actions_taken)
        self._done = (len(remaining) == 0) or (self._step >= self._task_cfg["max_steps"])

        return self._build_obs(), reward, self._done, {}

    def state(self) -> dict:
        return {
            "task":           self._task_name,
            "step":           self._step,
            "done":           self._done,
            "actions_taken":  {k: v.model_dump() for k, v in self._actions_taken.items()},
            "remaining_ids":  list(set(self._task_cfg["email_ids"]) - set(self._actions_taken)),
        }

    def close(self):
        pass

    # ── Reward logic ──────────────────────────────────────────────────────────

    def _compute_reward(self, email: dict, action: Action) -> Reward:
        correct_label = (action.label == email["true_label"])

        # Label reward (0.0, 0.5, or 1.0)
        if correct_label:
            label_score = 1.0
        elif self._is_close_label(action.label, email["true_label"]):
            label_score = 0.5
        else:
            label_score = 0.0

        # Reply reward (only matters for hard task or reply/urgent labels)
        reply_score = 0.0
        needs_reply = email["true_label"] in (Label.REPLY, Label.URGENT) and self._task_name == "hard"
        if needs_reply:
            if action.reply and len(action.reply.strip()) > 10:
                reply_text = action.reply.lower()
                hits = sum(1 for kw in email["reply_keywords"] if kw in reply_text)
                reply_score = min(1.0, hits / max(len(email["reply_keywords"]), 1))
            else:
                reply_score = 0.0
            total = round(0.6 * label_score + 0.4 * reply_score, 2)
            reason = (
                f"Label {'correct' if correct_label else 'incorrect'} "
                f"(+{0.6*label_score:.1f}), reply quality +{0.4*reply_score:.1f}"
            )
        else:
            total  = round(label_score, 2)
            reason = f"Label {'correct' if correct_label else 'wrong or close'} → {total}"

        return Reward(value=total, reason=reason)

    @staticmethod
    def _is_close_label(predicted: Label, true: Label) -> bool:
        """Near-misses: e.g. urgent vs reply are close; spam vs urgent are not."""
        close_pairs = {
            frozenset({Label.URGENT, Label.REPLY}),
            frozenset({Label.FOLLOW_UP, Label.ARCHIVE}),
        }
        return frozenset({predicted, true}) in close_pairs

    def _build_obs(self) -> Observation:
        ids    = self._task_cfg["email_ids"]
        emails = [
            {k: v for k, v in self._email_pool[eid].items()
             if k not in ("true_label", "reply_keywords")}
            for eid in ids
        ]
        return Observation(
            emails      = emails,
            step_number = self._step,
            task_hint   = self._task_cfg["hint"],
        )
