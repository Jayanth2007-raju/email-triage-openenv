"""
inference.py — OpenEnv Email Triage Baseline
Must be in root directory. Uses OpenAI client. Emits [START]/[STEP]/[END] logs.
"""

import os
import json
import sys
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "email-triage-openenv"
MAX_STEPS    = 8

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── Import env directly (same repo) ──────────────────────────────────────────
from env import EmailTriageEnv, Action, Label

VALID_LABELS = [l.value for l in Label]


def ask_llm(system_prompt: str, user_prompt: str) -> str:
    """Call the LLM and return its text response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=512,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def parse_action(llm_output: str, email_ids: list[int]) -> Action | None:
    """Parse the LLM's JSON output into an Action. Returns None on failure."""
    try:
        # Strip markdown code fences if present
        text = llm_output.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        data = json.loads(text)
        email_id = int(data["email_id"])
        label    = data["label"].strip().lower()
        reply    = data.get("reply", None)
        if label not in VALID_LABELS:
            return None
        if email_id not in email_ids:
            return None
        return Action(email_id=email_id, label=Label(label), reply=reply)
    except Exception:
        return None


SYSTEM_PROMPT = f"""You are an email triage assistant. Your job is to label emails.

Valid labels:
- urgent    → needs immediate attention (e.g. production issues, board meetings today)
- reply     → requires a written response (e.g. client requests, questions)
- follow_up → note to revisit later (e.g. deadlines weeks away, colleague requests)
- spam      → unwanted/promotional (delete/ignore)
- archive   → informational, no action needed (e.g. invoices, receipts)

You must respond with ONLY valid JSON in this exact format:
{{
  "email_id": <integer>,
  "label": "<one of: urgent, reply, follow_up, spam, archive>",
  "reply": "<draft reply text, or null if not applicable>"
}}

For emails labeled 'urgent' or 'reply', write a short professional reply draft.
For all other labels, set reply to null.
Do not include any explanation outside the JSON object."""


def run_task(task_name: str) -> tuple[float, list[float], int]:
    """Run one task episode. Returns (score, rewards_list, steps)."""
    env = EmailTriageEnv(task_name=task_name)
    obs = env.reset()

    task_cfg  = env._task_cfg
    email_ids = task_cfg["email_ids"]
    acted_ids: set[int] = set()
    rewards: list[float] = []
    step_n = 0
    done   = False
    last_error = None

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    while not done and step_n < MAX_STEPS:
        # Choose next email to act on
        remaining = [eid for eid in email_ids if eid not in acted_ids]
        if not remaining:
            break

        target_id = remaining[0]
        email     = next(e for e in obs.emails if e["id"] == target_id)

        user_prompt = (
            f"Task: {obs.task_hint}\n\n"
            f"Email to label:\n"
            f"ID: {email['id']}\n"
            f"From: {email['from']}\n"
            f"Subject: {email['subject']}\n"
            f"Body: {email['body']}\n\n"
            f"Respond with JSON only."
        )

        llm_out = ask_llm(SYSTEM_PROMPT, user_prompt)
        action  = parse_action(llm_out, email_ids)
        last_error = None

        if action is None:
            # Fallback: label as archive to avoid infinite loop
            action     = Action(email_id=target_id, label=Label.ARCHIVE, reply=None)
            last_error = "parse_error_fallback_to_archive"

        obs, reward, done, info = env.step(action)
        step_n += 1
        acted_ids.add(action.email_id)
        rewards.append(reward.value)

        error_str = last_error or info.get("error", info.get("warning", "null")) or "null"
        action_str = f"label(id={action.email_id}, label={action.label.value})"

        print(
            f"[STEP] step={step_n} action={action_str} "
            f"reward={reward.value:.2f} done={str(done).lower()} error={error_str}",
            flush=True
        )

    env.close()

    score    = round(sum(rewards) / len(rewards), 2) if rewards else 0.0
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success  = score >= 0.6

    print(
        f"[END] success={str(success).lower()} steps={step_n} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True
    )

    return score, rewards, step_n


def main():
    task_names = ["easy", "medium", "hard"]
    all_scores = []

    for task in task_names:
        try:
            score, rewards, steps = run_task(task)
            all_scores.append(score)
        except Exception as e:
            print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
            print(f"ERROR in task {task}: {e}", file=sys.stderr)

    if all_scores:
        avg = round(sum(all_scores) / len(all_scores), 2)
        print(f"\nOverall average score: {avg:.2f}", flush=True)


if __name__ == "__main__":
    main()
