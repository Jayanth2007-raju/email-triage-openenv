import os
import json
import requests
from openai import OpenAI

# ✅ SAFE ENV VARIABLES (FIXED)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BASE_URL = "https://22jayanth-email-triage-openenv.hf.space"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)


def get_action_from_llm(email, task):
    prompt = f"""
You are an email assistant.

Task: {task}

Email:
Subject: {email['subject']}
Body: {email['body']}

Choose label from: urgent, spam, archive, follow_up, reply.

Respond ONLY in JSON:
{{"label": "...", "reply": "..."}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = response.choices[0].message.content
    except Exception:
        text = '{"label": "archive", "reply": ""}'

    try:
        data = json.loads(text)
    except:
        data = {"label": "archive", "reply": ""}

    return data


def run_task(task_name):
    print(f"[START] task={task_name} env=email_triage model={MODEL_NAME}", flush=True)

    rewards = []
    success = False

    # RESET
    obs = requests.post(f"{BASE_URL}/reset", json={"task": task_name}).json()

    for step in range(1, 6):
        emails = obs["emails"]
        if not emails:
            break

        email = emails[0]

        # ✅ LLM CALL
        llm_output = get_action_from_llm(email, task_name)

        action = {
            "task": task_name,
            "email_id": email["id"],
            "label": llm_output.get("label", "archive"),
            "reply": llm_output.get("reply")
        }

        res = requests.post(f"{BASE_URL}/step", json=action).json()

        reward = float(res["reward"]["value"])
        done = res["done"]

        rewards.append(reward)

        print(
            f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
            flush=True
        )

        obs = res["observation"]

        if done:
            success = True
            break

    score = sum(rewards) / len(rewards) if rewards else 0.5

    # ✅ ensure (0,1)
    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={len(rewards)} score={score:.2f} rewards={rewards_str}",
        flush=True
    )


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)
