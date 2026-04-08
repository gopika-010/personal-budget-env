---
title: Personal Budget Env
emoji: 💰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
# PersonalBudgetEnv - OpenEnv Hackathon Submission

## Overview
**PersonalBudgetEnv** is a realistic personal finance simulation environment designed to train AI agents to manage monthly household budgeting like a real person in India.

The agent starts with a typical mid-level salary of ₹65,000, must handle recurring bills (Rent, Electricity, Internet), record daily expenses across categories, pay bills on time, and allocate money toward savings goals — all while staying solvent and making smart trade-offs.

**Key India-specific features:**
- UPI payment mode bonus
- Festival and medical expense tracking
- Realistic recurring bills and salary cycle
- Culturally relevant spending categories (food, transport, rent, etc.)

---

## Environment Description
This environment simulates **one month** of personal finance management. The agent receives dense rewards for good financial habits and penalties for overspending, delaying bills, or ignoring savings goals.

It provides a strong testbed for agentic reasoning, long-horizon planning, and prioritization.

---

## Action & Observation Spaces

### Actions (`BudgetAction`)
- `record_transaction` — Record daily spending (with category and UPI bonus)
- `pay_bill` — Pay recurring bills (Rent has priority bonus)
- `allocate_to_goal` — Move money to savings goals (emergency fund, vacation)
- `set_budget` — Adjust category spending limits
- `review_summary` — Check current financial status

### Observation (`BudgetObservation`)
- Current balance
- Monthly income (₹65,000)
- Category-wise budget (limit, spent, remaining)
- Upcoming unpaid bills
- Savings goal progress
- Recent transactions
- Human-readable inbox summary

---

## Tasks (Easy → Medium → Hard)

1. **Task 1 (Easy)**: Category Grader  
2. **Task 2 (Medium)**: Risk / Priority Grader  
3. **Task 3 (Hard)**: Financial Planning Grader  

---

## Evaluation Scores

- Task 1 (Category Grader): 1.0
- Task 2 (Risk Detection): 1.0
- Task 3 (Suggestion Quality): 0.85

**Total Reward:** 0.95

---

## Team
- **Gopika M**  – Environment Core & Lead
- **Teammate 1** – Inference Script
- **Teammate 2** – Server, Deployment & Documentation

---

## Setup & Usage

### Local Testing
```bash
pip install fastapi uvicorn pydantic requests openai
python -m uvicorn server:app --reload
