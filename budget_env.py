"""
PersonalBudgetEnv — Real-world personal finance environment for RL agents.

Simulates one month of household budgeting for a salaried individual in India.
The agent manages income, records expenses across categories, pays recurring bills,
and builds savings — all while staying solvent and making smart trade-offs.

India-specific mechanics:
  • UPI payment mode is rewarded (India's preferred digital payment method)
  • Festival and medical expenses are explicitly tracked (culturally relevant)
  • Salary of ₹65,000/month credited at episode start (realistic mid-level income)
  • Categories match real Indian household spending patterns (INR)

OpenEnv interface:
  reset()        → BudgetObservation   (fresh episode, salary credited)
  state()        → BudgetObservation   (current observable state)
  step(action)   → (BudgetObservation, BudgetReward, done: bool, info: dict)

Tasks (easy → medium → hard):
  task1_easy_category_grader      — correct category per transaction
  task2_medium_risk_grader        — detect and respond to high-risk states
  task3_hard_suggestion_grader    — useful corrective actions + balance + goals
"""

import random
from typing import Tuple, Dict, Any, List

from models import (
    BudgetAction,
    BudgetObservation,
    BudgetReward,
    CategoryBudget,
    Bill,
    Goal,
    Transaction,
)


class PersonalBudgetEnvironment:
    # ─────────────────────────────────────────
    #  Constants
    # ─────────────────────────────────────────

    MONTHLY_INCOME: float = 65_000.0  # INR

    DEFAULT_BUDGET_LIMITS: Dict[str, float] = {
        "food":          8_000.0,
        "rent":         15_000.0,
        "transport":     4_000.0,
        "utilities":     3_000.0,
        "savings":      15_000.0,
        "entertainment": 3_000.0,
        "medical":       2_000.0,
        "festival":      5_000.0,
    }

    BILL_TEMPLATES: List[Dict[str, Any]] = [
        {"name": "Electricity", "amount": 2_500.0,  "category": "utilities"},
        {"name": "Internet",    "amount":   999.0,  "category": "utilities"},
        {"name": "Rent",        "amount": 15_000.0, "category": "rent"},
    ]

    VALID_CATEGORIES = set(DEFAULT_BUDGET_LIMITS.keys())
    MAX_STEPS: int = 30

    # ─────────────────────────────────────────
    #  Init
    # ─────────────────────────────────────────

    def __init__(self):
        self.balance: float = 0.0
        self.category_budgets: Dict[str, Dict[str, float]] = {}
        self.transactions: List[Dict[str, Any]] = []
        self.bills: List[Dict[str, Any]] = []
        self.goals: Dict[str, Dict[str, float]] = {}
        self.step_count: int = 0
        self.done: bool = False
        self.risk_events: List[Dict[str, Any]] = []
        self._prev_was_high_risk: bool = False
        self.reset()

    # ─────────────────────────────────────────
    #  OpenEnv API: reset()
    # ─────────────────────────────────────────

    def reset(self) -> BudgetObservation:
        """
        Start a fresh episode.
        Sets balance to monthly salary, resets all bills to unpaid,
        zeros all spending, and clears transaction history.
        random.seed(42) makes grader scores reproducible.
        """
        random.seed(42)

        self.balance    = self.MONTHLY_INCOME
        self.step_count = 0
        self.done       = False

        # Fresh category budgets — zero spending
        self.category_budgets = {
            cat: {"limit": lim, "spent": 0.0}
            for cat, lim in self.DEFAULT_BUDGET_LIMITS.items()
        }

        # All bills start unpaid
        self.bills = [
            {**tpl, "paid": False}
            for tpl in self.BILL_TEMPLATES
        ]

        # Savings goals reset to zero progress
        self.goals = {
            "emergency_fund": {"target": 10_000.0, "current": 0.0},
            "vacation":       {"target":  5_000.0, "current": 0.0},
        }

        self.transactions         = []
        self.risk_events          = []
        self._prev_was_high_risk  = False

        return self.state()

    # ─────────────────────────────────────────
    #  OpenEnv API: state()
    # ─────────────────────────────────────────

    def state(self) -> BudgetObservation:
        """
        Return the full observable state as a validated Pydantic model.
        Safe to call at any point without mutating state.
        """
        category_view = {
            cat: CategoryBudget(
                limit=data["limit"],
                spent=round(data["spent"], 2),
                remaining=round(data["limit"] - data["spent"], 2),
            )
            for cat, data in self.category_budgets.items()
        }

        unpaid_bills = [
            Bill(
                name=b["name"],
                amount=b["amount"],
                category=b["category"],
                paid=b["paid"],
            )
            for b in self.bills if not b["paid"]
        ]

        goal_view = {
            name: Goal(target=g["target"], current=round(g["current"], 2))
            for name, g in self.goals.items()
        }

        recent_txns = [
            Transaction(
                type=t["type"],
                amount=t["amount"],
                category=t["category"],
                description=t.get("description", ""),
                mode=t["mode"],
            )
            for t in self.transactions[-5:]
        ]

        # Human-readable status string
        unpaid_names = [b["name"] for b in self.bills if not b["paid"]]
        over_cats    = [c for c, d in self.category_budgets.items() if d["spent"] > d["limit"]]

        parts = [f"Balance: ₹{self.balance:,.0f}"]
        if unpaid_names:
            parts.append(f"Unpaid: {', '.join(unpaid_names)}")
        if over_cats:
            parts.append(f"Over budget: {', '.join(over_cats)}")
        if not unpaid_names and not over_cats:
            parts.append("All bills paid — no overspends")

        return BudgetObservation(
            current_balance=round(self.balance, 2),
            monthly_income=self.MONTHLY_INCOME,
            category_budgets=category_view,
            recent_transactions=recent_txns,
            upcoming_bills=unpaid_bills,
            goals=goal_view,
            step_count=self.step_count,
            inbox_summary=" | ".join(parts),
        )

    # ─────────────────────────────────────────
    #  OpenEnv API: step()
    # ─────────────────────────────────────────

    def step(
        self, action: BudgetAction
    ) -> Tuple[BudgetObservation, BudgetReward, bool, Dict[str, Any]]:
        """
        Apply the agent's action and return (observation, reward, done, info).

        done=True when:
          • step_count reaches MAX_STEPS (natural episode end), or
          • balance drops below 0 (agent went bankrupt — early termination)
        """
        if self.done:
            return (
                self.state(),
                BudgetReward(value=0.0, reason="episode already ended"),
                True,
                {"task_scores": self._get_task_scores()},
            )

        self.step_count += 1
        feedback = self._apply_action(action)
        reward   = self._calculate_reward(action, feedback)

        # ── Risk event tracking ─────────────────────────────────────────
        if self._prev_was_high_risk:
            useful_action = action.action_type in (
                "pay_bill", "allocate_to_goal", "set_budget", "review_summary"
            )
            self.risk_events.append({
                "step":          self.step_count,
                "detected":      action.action_type != "record_transaction",
                "useful_action": useful_action,
            })
        self._prev_was_high_risk = self._is_high_risk()
        # ───────────────────────────────────────────────────────────────

        self.done = (self.step_count >= self.MAX_STEPS) or (self.balance < 0)

        info: Dict[str, Any] = {
            "task_scores":     self._get_task_scores(),
            "action_feedback": feedback,
            "step":            self.step_count,
            "done":            self.done,
        }

        return self.state(), reward, self.done, info

    # ─────────────────────────────────────────
    #  Action application
    # ─────────────────────────────────────────

    def _apply_action(self, action: BudgetAction) -> Dict[str, Any]:
        """Mutate environment state based on the action."""
        fb: Dict[str, Any] = {
            "valid":              True,
            "message":            "",
            "over_budget":        False,
            "unknown_category":   False,
            "bill_already_paid":  False,
            "unknown_bill":       False,
            "unknown_goal":       False,
            "goal_completed":     False,
            "insufficient_funds": False,
        }

        # ── record_transaction ─────────────────────────────────────────
        if action.action_type == "record_transaction":
            if action.amount <= 0:
                fb["valid"]   = False
                fb["message"] = "Amount must be > 0"
                return fb

            self.balance -= action.amount

            if action.category not in self.VALID_CATEGORIES:
                fb["unknown_category"] = True
                fb["message"]          = f"Unknown category '{action.category}'"
            else:
                self.category_budgets[action.category]["spent"] += action.amount
                remaining = (
                    self.category_budgets[action.category]["limit"]
                    - self.category_budgets[action.category]["spent"]
                )
                if remaining < 0:
                    fb["over_budget"] = True
                    fb["message"]     = (
                        f"Over budget in '{action.category}' by ₹{abs(remaining):,.0f}"
                    )

            self.transactions.append({
                "type":        "expense",
                "amount":      action.amount,
                "category":    action.category,
                "description": action.description,
                "mode":        action.mode,
            })
            fb["message"] = fb["message"] or "Transaction recorded"

        # ── set_budget ─────────────────────────────────────────────────
        elif action.action_type == "set_budget":
            if action.category not in self.VALID_CATEGORIES:
                fb["valid"]            = False
                fb["unknown_category"] = True
                fb["message"]          = f"Unknown category '{action.category}'"
                return fb

            if action.target <= 0:
                fb["valid"]   = False
                fb["message"] = "Budget target must be > 0"
                return fb

            old = self.category_budgets[action.category]["limit"]
            self.category_budgets[action.category]["limit"] = action.target
            fb["message"] = (
                f"'{action.category}' limit: ₹{old:,.0f} → ₹{action.target:,.0f}"
            )

        # ── pay_bill ───────────────────────────────────────────────────
        elif action.action_type == "pay_bill":
            bill = next(
                (b for b in self.bills
                 if b["name"].lower() == action.bill_name.lower()),
                None,
            )
            if bill is None:
                fb["valid"]        = False
                fb["unknown_bill"] = True
                fb["message"]      = f"No bill named '{action.bill_name}'"
                return fb

            if bill["paid"]:
                fb["bill_already_paid"] = True
                fb["message"]           = f"'{bill['name']}' already paid"
                return fb

            if self.balance < bill["amount"]:
                fb["valid"]              = False
                fb["insufficient_funds"] = True
                fb["message"]            = (
                    f"Insufficient funds to pay '{bill['name']}' "
                    f"(need ₹{bill['amount']:,.0f}, have ₹{self.balance:,.0f})"
                )
                return fb

            bill["paid"]  = True
            self.balance -= bill["amount"]
            cat           = bill["category"]

            if "rent" in bill["name"].lower():
                fb["high_priority_bill_paid"] = True
            elif "electricity" in bill["name"].lower() or "internet" in bill["name"].lower():
                fb["medium_priority_bill_paid"] = True

            if cat in self.category_budgets:
                self.category_budgets[cat]["spent"] += bill["amount"]

            self.transactions.append({
                "type":        "bill_payment",
                "amount":      bill["amount"],
                "category":    cat,
                "description": f"{bill['name']} bill",
                "mode":        action.mode,
            })
            fb["message"] = f"Paid '{bill['name']}': ₹{bill['amount']:,.0f}"

        # ── allocate_to_goal ───────────────────────────────────────────
        elif action.action_type == "allocate_to_goal":
            if action.goal_name not in self.goals:
                fb["valid"]        = False
                fb["unknown_goal"] = True
                fb["message"]      = f"Unknown goal '{action.goal_name}'"
                return fb

            if action.amount <= 0:
                fb["valid"]   = False
                fb["message"] = "Allocation amount must be > 0"
                return fb

            if action.amount > self.balance:
                fb["valid"]              = False
                fb["insufficient_funds"] = True
                fb["message"]            = "Insufficient balance for goal allocation"
                return fb

            goal        = self.goals[action.goal_name]
            to_allocate = min(action.amount, goal["target"] - goal["current"])

            # Nothing left to allocate — goal already complete
            if to_allocate <= 0:
                fb["message"] = f"Goal '{action.goal_name}' already complete"
                return fb

            goal["current"] = round(goal["current"] + to_allocate, 2)
            self.balance    = round(self.balance - to_allocate, 2)

            if goal["current"] >= goal["target"]:
                fb["goal_completed"] = True
                fb["message"]        = f"Goal '{action.goal_name}' COMPLETED! 🎯"
            else:
                pct = (goal["current"] / goal["target"]) * 100
                fb["message"] = (
                    f"Allocated ₹{to_allocate:,.0f} to '{action.goal_name}' "
                    f"({pct:.0f}% of ₹{goal['target']:,.0f})"
                )

        # ── review_summary ─────────────────────────────────────────────
        elif action.action_type == "review_summary":
            fb["message"] = (
                f"Reviewed summary at step {self.step_count} | "
                f"Balance: ₹{self.balance:,.0f}"
            )

        return fb

    # ─────────────────────────────────────────
    #  Reward function
    # ─────────────────────────────────────────

    def _calculate_reward(
        self,
        action: BudgetAction,
        feedback: Dict[str, Any],
    ) -> BudgetReward:
        score   = 0.0
        reasons: List[str] = []

        # Invalid actions
        if not feedback.get("valid", True):
            if feedback.get("insufficient_funds"):
                score -= 0.25
                reasons.append("insufficient funds -0.25")
            else:
                score -= 0.35
                reasons.append("invalid action -0.35")
            return BudgetReward(
                value=round(max(0.0, min(1.0, score)), 3),
                reason=" | ".join(reasons),
            )

        # === Per-action rewards ===
        if action.action_type == "record_transaction":
            score += 0.20
            reasons.append("transaction logged +0.20")

            if action.mode == "UPI":
                score += 0.05
                reasons.append("UPI bonus +0.05")

            if action.description and len(action.description.strip()) > 5:
                score += 0.05
                reasons.append("good description +0.05")

            if action.category in ("festival", "medical"):
                score += 0.08
                reasons.append(f"{action.category} tracked +0.08")

            if feedback.get("over_budget"):
                score -= 0.40
                reasons.append("over budget -0.40")

            if feedback.get("unknown_category"):
                score -= 0.18
                reasons.append("uncategorised spend -0.18")

        elif action.action_type == "set_budget":
            score += 0.18
            reasons.append("budget adjusted +0.18")

        elif action.action_type == "pay_bill":
            if feedback.get("bill_already_paid"):
                score -= 0.05
                reasons.append("duplicate payment -0.05")
            else:
                score += 0.38
                reasons.append("bill paid +0.38")

                if feedback.get("high_priority_bill_paid"):
                    score += 0.15
                    reasons.append("rent priority +0.15")
                elif feedback.get("medium_priority_bill_paid"):
                    score += 0.07
                    reasons.append("utility priority +0.07")

                if action.mode == "UPI":
                    score += 0.05
                    reasons.append("UPI bill +0.05")

        elif action.action_type == "allocate_to_goal":
            score += 0.32
            reasons.append("goal allocation +0.32")
            if feedback.get("goal_completed"):
                score += 0.20
                reasons.append("goal completed +0.20")

        elif action.action_type == "review_summary":
            score += 0.08
            reasons.append("review check-in +0.08")

        # === Global state adjustments (every step) ===
        balance_ratio = max(0.0, self.balance / self.MONTHLY_INCOME)
        balance_bonus = round(0.08 * min(balance_ratio, 1.0), 4)
        score        += balance_bonus
        reasons.append(f"balance safety +{balance_bonus:.2f}")

        if self.balance < 0:
            score -= 0.60
            reasons.append("negative balance -0.60")

        # Unpaid bill penalty — light early, strong late
        unpaid_count = sum(1 for b in self.bills if not b.get("paid", True))
        if unpaid_count > 0 and self.step_count > 2:
            penalty = (
                round(0.02 * unpaid_count * self.step_count, 4)
                if self.step_count > 3
                else round(0.04 * unpaid_count, 4)
            )
            score  -= penalty
            reasons.append(f"unpaid bills -{penalty:.2f}")

        # Goal progress penalty after mid-episode
        if self.step_count > 15:
            total_progress = (
                sum(
                    g["current"] / g["target"]
                    for g in self.goals.values()
                    if g["target"] > 0
                ) / len(self.goals)
            ) if self.goals else 0.0

            if total_progress < 0.25:
                score -= 0.10
                reasons.append("low goal progress -0.10")

        # Spam penalty for review_summary
        if action.action_type == "review_summary" and self.step_count > 8:
            score -= 0.04
            reasons.append("repeated review -0.04")

        final_score = round(max(0.0, min(1.0, score)), 3)
        return BudgetReward(
            value=final_score,
            reason=" | ".join(reasons) or "neutral",
        )

    # ─────────────────────────────────────────
    #  Task graders  (deterministic, 0.0–1.0)
    # ─────────────────────────────────────────

    def _get_task_scores(self) -> Dict[str, float]:
        """
        Returns all three task scores clamped to [0.0, 1.0].
        All three keys must exactly match get_tasks() IDs.
        """
        def safe(x: float) -> float:
            return round(max(0.0, min(1.0, x)), 4)

        return {
            "task1_easy_category_grader":   safe(self._grade_task1()),
            "task2_medium_risk_grader":     safe(self._grade_task2()),
            "task3_hard_suggestion_grader": safe(self._grade_task3()),
        }

    def get_tasks(self) -> List[Dict[str, Any]]:
        """Returns task metadata — IDs must exactly match _get_task_scores() keys."""
        return [
            {
                "id":          "task1_easy_category_grader",
                "name":        "Category Grader",
                "difficulty":  "easy",
                "description": (
                    "Check whether the agent assigns the correct spending category "
                    "to each transaction. Example: '₹500 spent on Swiggy' → 'food'. "
                    "Score = correct categorisations / total transactions."
                ),
            },
            {
                "id":          "task2_medium_risk_grader",
                "name":        "Risk / Priority Grader",
                "difficulty":  "medium",
                "description": (
                    "Check whether the agent correctly detects dangerous financial "
                    "situations (low balance, overspend, unpaid bills) and flags them "
                    "as high-risk. Score = risk situations detected and acted upon."
                ),
            },
            {
                "id":          "task3_hard_suggestion_grader",
                "name":        "Suggestion Grader",
                "difficulty":  "hard",
                "description": (
                    "Check whether the agent's corrective actions are useful. "
                    "After a high-risk event the agent should reduce spending, "
                    "pay bills, or allocate to savings — not keep spending freely. "
                    "Score = useful corrective actions / total risk events."
                ),
            },
        ]

    def run_task(self, task_id: str) -> Dict[str, Any]:
        """Run a single task and return its score."""
        self.reset()
        score_map = {
            "task1_easy_category_grader":   self._grade_task1,
            "task2_medium_risk_grader":     self._grade_task2,
            "task3_hard_suggestion_grader": self._grade_task3,
        }
        if task_id not in score_map:
            raise ValueError(f"Unknown task_id: {task_id}")
        return {
            "task_id": task_id,
            "score":   round(score_map[task_id](), 4),
        }

    # ── Grader implementations ─────────────────────────────────────────

    CATEGORY_KEYWORDS: Dict[str, List[str]] = {
        "food":          ["swiggy", "zomato", "restaurant", "lunch", "dinner",
                          "breakfast", "grocery", "groceries", "blinkit", "zepto",
                          "chai", "canteen", "mess", "hotel", "cafe"],
        "transport":     ["ola", "uber", "auto", "metro", "bus", "petrol",
                          "fuel", "cab", "rapido", "train", "irctc"],
        "utilities":     ["electricity", "internet", "wifi", "airtel", "jio",
                          "bsnl", "water", "gas", "lpg", "recharge"],
        "rent":          ["rent", "pg", "hostel", "accommodation", "landlord"],
        "entertainment": ["netflix", "hotstar", "prime", "movie", "concert",
                          "spotify", "game", "shopping", "amazon", "flipkart"],
        "medical":       ["doctor", "hospital", "pharmacy", "medicine", "clinic",
                          "apollo", "health", "dental", "chemist"],
        "festival":      ["diwali", "puja", "eid", "christmas", "holi",
                          "festival", "pooja", "temple", "donation", "gift"],
        "savings":       ["savings", "fd", "sip", "mutual fund", "investment",
                          "ppf", "emergency"],
    }

    def _infer_expected_category(self, description: str) -> str:
        """Return the most likely category for a transaction description."""
        desc_lower = description.lower()
        for cat, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in desc_lower for kw in keywords):
                return cat
        return ""  # ambiguous — skip in grader

    def _grade_task1(self) -> float:
        """
        Task 1 (Easy) — Category Grader.
        score = correct_categorisations / categorisable_transactions
        Ambiguous descriptions are skipped (not penalised).
        """
        categorisable = 0
        correct       = 0

        for txn in self.transactions:
            desc     = txn.get("description", "")
            expected = self._infer_expected_category(desc)
            if not expected:
                continue  # ambiguous — skip

            categorisable += 1
            if txn.get("category", "") == expected:
                correct += 1

        if categorisable == 0:
            # Partial credit if at least something was logged
            return 0.5 if self.transactions else 0.1

        return correct / categorisable

    def _is_high_risk(self) -> bool:
        """Detect whether the current state is financially high-risk."""
        low_balance = self.balance < (self.MONTHLY_INCOME * 0.18)
        overspent   = any(
            d["spent"] > d["limit"] * 1.12
            for d in self.category_budgets.values()
        )
        late_bills  = self.step_count > 12 and any(
            not b.get("paid", True) for b in self.bills
        )
        return low_balance or overspent or late_bills

    def _grade_task2(self) -> float:
        """
        Task 2 (Medium) — Risk / Priority Grader.
        score = detected_risk_events / total_risk_events
        A risk event is detected when the agent's next action is NOT another spend.
        """
        if not self.risk_events:
            # No risk events — episode was well-managed, award near-full score
            return 0.9

        detected = sum(1 for e in self.risk_events if e["detected"])
        return detected / len(self.risk_events)

    def _grade_task3(self) -> float:
        """
        Task 3 (Hard) — Suggestion Grader.

        score = 0.60 × (useful corrections / risk events)
              + 0.25 × balance health
              + 0.15 × goal completion bonus

        Useful actions after high-risk: pay_bill, allocate_to_goal,
        set_budget, review_summary. Continuing to spend is not useful.
        """
        # Component 1: useful corrective actions
        if self.risk_events:
            useful    = sum(1 for e in self.risk_events if e["useful_action"])
            sug_score = useful / len(self.risk_events)
        else:
            sug_score = 0.9  # no risk events — episode was healthy

        # Component 2: ending balance health (capped at 1.0)
        health_target = self.MONTHLY_INCOME * 0.20
        balance_score = min(max(self.balance, 0.0) / health_target, 1.0)

        # Component 3: savings goal completion (capped at 1.0)
        completed_goals = sum(
            1 for g in self.goals.values()
            if g["current"] >= g["target"]
        )
        goal_bonus = (
            min(completed_goals / len(self.goals), 1.0)
            if self.goals else 0.0
        )

        raw = (sug_score * 0.60) + (balance_score * 0.25) + (goal_bonus * 0.15)
        return round(max(0.0, min(1.0, raw)), 4)