from typing import List


class RetrievalAgent:
    def plan(self, question: str, top_k: int) -> tuple[int, List[str]]:
        trace: List[str] = ["planner: analyzed query intent"]
        q = question.lower()

        tuned_top_k = top_k
        if any(k in q for k in ["summarize", "summary", "overall", "all"]):
            tuned_top_k = min(10, max(top_k, 7))
            trace.append(f"planner: broad query detected, increased top_k to {tuned_top_k}")
        else:
            trace.append(f"planner: focused query, keeping top_k={tuned_top_k}")

        return tuned_top_k, trace
