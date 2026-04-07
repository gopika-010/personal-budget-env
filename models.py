from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any
class BudgetAction(BaseModel):
    action_type: Literal[
        "record_transaction",
        "set_budget",
        "pay_bill",
        "allocate_to_goal",
        "review_summary"
    ]
    amount: float = 0.0
    target: float = 0.0
    category: str = ""
    description: str = ""
    mode: Literal["UPI", "cash", "card"] = "UPI"
    bill_name: str = ""
    goal_name: str = ""


class CategoryBudget(BaseModel):
    limit: float
    spent: float
    remaining: float


class Bill(BaseModel):
    name: str
    amount: float
    category: str
    paid: bool


class Goal(BaseModel):
    target: float
    current: float


class Transaction(BaseModel):
    type: str
    amount: float
    category: str
    description: str = ""
    mode: str


class BudgetObservation(BaseModel):
    current_balance: float
    monthly_income: float
    category_budgets: Dict[str, CategoryBudget]
    recent_transactions: List[Transaction]
    upcoming_bills: List[Bill]
    goals: Dict[str, Goal]
    step_count: int
    inbox_summary: str


class BudgetReward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    reason: str