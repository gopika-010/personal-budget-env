"""
inference.py — PersonalBudgetEnv baseline agent
------------------------------------------------
- Calls the environment via HTTP (not direct import)
- Uses OpenAI client with API_BASE_URL + MODEL_NAME + HF_TOKEN
- LLM makes all decisions (no heavy rule-based override)
- Strict [START] / [STEP] / [END] log format required by judges
- 3 separate tasks with individual scores in [END]
- Total runtime < 20 min on 2vCPU / 8GB machine
"""

import os
import json
import requests
from typing import List, Dict, Any
from openai import OpenAI


API_BASE_URL  = os.getenv(
    "API_BASE_URL",
    "https://sweathabala-personal-budget-env.hf.space"
)

MODEL_NAME    = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN      = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", HF_TOKEN)

llm_client = OpenAI(
    api_key=OPENAI_API_KEY or HF_TOKEN,
    base_url=os.getenv("LLM_BASE_URL", "https://api-inference.huggingface.co/v1"),
)

ENV_URL = API_BASE_URL.rstrip("/")


def env_reset() -> Dict[str, Any]:
    try:
        resp = requests.post(
            f"{ENV_URL}/reset",
            headers={"Content-Type": "application/json"},
            json={},  # important
            timeout=60
        )
        print("STATUS:", resp.status_code, flush=True)
        resp.raise_for_status()
        return resp.json().get("observation", {})
    except Exception as e:
        print(f"[ERROR] env_reset failed: {e}", flush=True)
        return {
            "step_count": 0,
            "current_balance": 10000,
            "monthly_income": 50000,
            "upcoming_bills": [],
            "goals": {},
            "category_budgets": {},
            "inbox_summary": "fallback"
        }

def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    try:
        resp = requests.post(
            f"{ENV_URL}/step",
            headers={"Content-Type": "application/json"},
            json=action,
            timeout=60
        )
        print("STEP STATUS:", resp.status_code, flush=True)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[ERROR] env_step failed: {e}", flush=True)
        return {
            "observation": {},
            "reward": {"value": 0.0, "reason": "fallback"},
            "done": True,
            "info": {}
        }

SYSTEM_PROMPT = """You are a personal finance assistant managing a monthly budget in India.

Your job each step: decide ONE action to take based on the current financial state.

PRIORITIES (in order):
1. Pay any unpaid bills FIRST
2. Allocate money to savings goals
3. Record daily transactions
4. Use review_summary if unsure

VALID action_types:
pay_bill | allocate_to_goal | record_transaction | set_budget | review_summary

You MUST respond with ONLY valid JSON.
"""

def format_observation(obs: Dict[str, Any]) -> str:
    return f"""
Step: {obs.get('step_count', 0)}
Balance: ₹{obs.get('current_balance', 0)}
Income: ₹{obs.get('monthly_income', 0)}
"""

def call_llm(history: List[Dict]) -> Dict[str, Any]:
    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            max_tokens=150,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]

        return json.loads(raw.strip())
    except Exception:
        return {"action_type": "review_summary"}

def validate_action(action: Dict[str, Any]) -> Dict[str, Any]:
    valid_types = {
        "pay_bill", "allocate_to_goal",
        "record_transaction", "set_budget", "review_summary"
    }

    if action.get("action_type") not in valid_types:
        action["action_type"] = "review_summary"

    action.setdefault("amount", 0.0)
    action.setdefault("mode", "UPI")
    action.setdefault("category", "")
    action.setdefault("description", "")
    action.setdefault("bill_name", "")
    action.setdefault("goal_name", "")

    return action


def run_task(task_id: str, task_index: int) -> Dict[str, Any]:
    obs = env_reset()
    history = []
    total_reward = 0.0

    for step in range(30):
        obs_text = format_observation(obs)
        history.append({"role": "user", "content": obs_text})

        action = validate_action(call_llm(history[-6:]))

        history.append({"role": "assistant", "content": json.dumps(action)})

        result = env_step(action)

        obs = result.get("observation", {})
        reward = result.get("reward", {}).get("value", 0.0)
        reason = result.get("reward", {}).get("reason", "")
        done = result.get("done", True)

        total_reward += reward

        print(f"[STEP] {json.dumps({'task':task_id,'step':step,'action':action['action_type'],'reward':reward,'reason':reason})}")

        if done:
            break

    return {
        "task_id": task_id,
        "total_reward": total_reward,
        "task_scores": {}
    }


def main():
    tasks = [
        "task1_easy_category_grader",
        "task2_medium_risk_grader",
        "task3_hard_suggestion_grader",
    ]

    print(f'[START] {json.dumps({"task_id": "all", "model": MODEL_NAME})}')

    all_scores = {}

    for i, task_id in enumerate(tasks):
        result = run_task(task_id, i)
        all_scores[task_id] = result["total_reward"]

    print(f'[END] {json.dumps({"total_score": sum(all_scores.values()), "task_scores": all_scores})}')


if __name__ == "__main__":
    main()