import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from budget_env import PersonalBudgetEnvironment
from models import BudgetAction
import uvicorn

app = FastAPI(title="PersonalBudgetEnv - OpenEnv Hackathon")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = PersonalBudgetEnvironment()

@app.get("/")
def home():
    return {
        "name": "PersonalBudgetEnv",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"],
        "status": "running",
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {"status": "ok", "env": "PersonalBudgetEnv", "version": "1.0.0"}

@app.post("/reset")
async def reset():
    observation = env.reset()
    return {"observation": observation.model_dump()}

@app.post("/step")
async def step(action: BudgetAction):
    observation, reward, done, info = env.step(action)
    return {
        "observation": observation.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }

@app.get("/state")
async def get_state():
    return {"state": env.state().model_dump()}

@app.get("/tasks")
def tasks():
    return {"tasks": env.get_tasks()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)