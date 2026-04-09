from env import EmailTriageEnv, Action


def safe_score(total: float, n: int) -> float:
    """
    Ensure score is strictly between (0,1)
    """
    if n == 0:
        return 0.5  # safe default

    score = round(total / n, 4)

    if score <= 0.0:
        return 0.01
    elif score >= 1.0:
        return 0.99

    return score


def grade_easy(actions: list[dict]) -> float:
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

    return safe_score(total, n)


def grade_medium(actions: list[dict]) -> float:
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

    return safe_score(total, n)


def grade_hard(actions: list[dict]) -> float:
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

    return safe_score(total, n)


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}
