from fastapi import FastAPI
from budget_env import PersonalBudgetEnvironment   
from models import BudgetAction                   
import uvicorn

app = FastAPI(title="PersonalBudgetEnv - OpenEnv Hackathon")

@app.get("/")
def home():
    return {
        "status": "running",
        "docs": "/docs",
        "message": "Open /docs to test API"
    }


env = PersonalBudgetEnvironment()

@app.post("/reset")
async def reset():
    """Reset the environment and return initial observation."""
    observation = env.reset()
    return {"observation": observation.model_dump()}

@app.post("/step")
async def step(action: BudgetAction):
    """Take one action and return observation, reward, done, info."""
    observation, reward, done, info = env.step(action)
    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }

@app.get("/state")
async def get_state():
    """Return current state (for debugging)."""
    return {"state": env.state().model_dump()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
