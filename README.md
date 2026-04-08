# Email Triage OpenEnv

An OpenEnv environment where an AI agent triages a realistic inbox: classifying emails and drafting replies.

## Why this domain?

Email triage is one of the most universal productivity tasks. Every knowledge worker does it daily. A capable agent must understand context, urgency, sender intent, and appropriate response tone — making it an excellent RL/evaluation benchmark.

## Environment description

The agent receives a set of emails and must assign each one a label and (for urgent/reply emails in the hard task) write a draft reply.

## Action space

| Field | Type | Values |
|---|---|---|
| `email_id` | integer | ID of email to act on |
| `label` | enum | `urgent`, `reply`, `follow_up`, `spam`, `archive` |
| `reply` | string or null | Draft reply text (required for `urgent`/`reply` in hard task) |

## Observation space

| Field | Type | Description |
|---|---|---|
| `emails` | list[dict] | Each email: `id`, `from`, `subject`, `body` |
| `step_number` | integer | Current step |
| `task_hint` | string | Natural language instruction |

## Tasks

| Task | Difficulty | Emails | Max steps | Description |
|---|---|---|---|---|
| `easy` | Easy | 3 | 3 | Label clearly distinct emails (spam, urgent, archive) |
| `medium` | Medium | 5 | 5 | Label 5 emails including ambiguous categories |
| `hard` | Hard | 8 | 8 | Label all 8 emails + draft replies for urgent/reply |

## Reward function

- **Label correct**: +1.0
- **Label is close miss** (e.g. urgent vs reply, follow_up vs archive): +0.5
- **Label wrong**: 0.0
- **Hard task only**: reply quality contributes 40% of score for `urgent`/`reply` emails (based on keyword coverage)

## Setup

```bash
pip install -r requirements.txt
python app.py          # starts server on port 7860
```

## Docker

```bash
docker build -t email-triage-openenv .
docker run -p 7860:7860 email-triage-openenv
```

## API usage

```bash
# Reset
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task":"medium"}'

# Step
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" \
  -d '{"task":"medium","email_id":1,"label":"urgent","reply":"I will attend the 3pm meeting."}'

# State
curl http://localhost:7860/state?task=medium
```

## Run inference baseline

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## Baseline scores (Qwen2.5-72B-Instruct)

| Task | Score |
|---|---|
| easy | ~0.90 |
| medium | ~0.75 |
| hard | ~0.60 |

## Hugging Face Space

Deploy by pushing this repo to a HF Space with SDK: docker and the `openenv` tag.
