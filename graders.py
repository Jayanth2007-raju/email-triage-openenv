"""
Deterministic graders for all 3 tasks.
Each grader returns a float in [0.0, 1.0].
"""

from env import EmailTriageEnv, Action, Label, TASKS, EMAILS


def _run_episode(task_name: str, actions: list[dict]) -> float:
    """Helper: replay a fixed action list and return mean reward."""
    env = EmailTriageEnv(task_name=task_name)
    env.reset()
    total, n = 0.0, 0
    for act in actions:
        action = Action(**act)
        _, reward, done, _ = env.step(action)
        total += reward.value
        n += 1
        if done:
            break
    return round(total / n, 4) if n else 0.0


def grade_easy(actions: list[dict]) -> float:
    """
    Easy task: label 3 emails (spam, urgent, archive).
    Perfect score = 1.0, one wrong = ~0.67, all wrong = 0.0
    """
    env = EmailTriageEnv(task_name="easy")
    env.reset()
    total, n = 0.0, 0
    for act in actions:
        action = Action(**act)
        _, reward, done, _ = env.step(action)
        total += reward.value
        n += 1
        if done:
            break
    return round(total / n, 4) if n else 0.0


def grade_medium(actions: list[dict]) -> float:
    """
    Medium task: label 5 emails including ambiguous ones.
    """
    env = EmailTriageEnv(task_name="medium")
    env.reset()
    total, n = 0.0, 0
    for act in actions:
        action = Action(**act)
        _, reward, done, _ = env.step(action)
        total += reward.value
        n += 1
        if done:
            break
    return round(total / n, 4) if n else 0.0


def grade_hard(actions: list[dict]) -> float:
    """
    Hard task: label 8 emails AND draft replies for urgent/reply emails.
    Reply quality affects 40% of those emails' scores.
    """
    env = EmailTriageEnv(task_name="hard")
    env.reset()
    total, n = 0.0, 0
    for act in actions:
        action = Action(**act)
        _, reward, done, _ = env.step(action)
        total += reward.value
        n += 1
        if done:
            break
    return round(total / n, 4) if n else 0.0


GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}
