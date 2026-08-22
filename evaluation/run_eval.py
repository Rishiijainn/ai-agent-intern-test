import json
from pathlib import Path
from typing import List, Dict, Any
from src.agent import AsterRowAgent
from src.config import EVALUATION_DIR

def load_cases(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("cases", [])
    except Exception as err:
        print(f"[Error] Failed loading {file_path}: {err}")
    return []

def run_evaluation_suite():
    agent = AsterRowAgent()
    visible_file = EVALUATION_DIR / "visible-cases.json"
    custom_file = EVALUATION_DIR / "custom-cases.json"

    visible_cases = load_cases(visible_file)
    custom_cases = load_cases(custom_file)
    all_cases = visible_cases + custom_cases

    print(f"\n=======================================================")
    print(f"       ASTER & ROW AI AGENT BENCHMARK SUITE")
    print(f"       Loaded: {len(visible_cases)} visible + {len(custom_cases)} custom = {len(all_cases)} total")
    print(f"=======================================================\n")

    if not all_cases:
        print("No test cases found. Please check evaluation/ folder.")
        return

    category_stats = {}
    passed_count = 0

    for idx, case in enumerate(all_cases, 1):
        case_id = case.get("id", f"case_{idx}")
        category = case.get("category", "general")
        
        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0}
        category_stats[category]["total"] += 1

        # Extract user messages
        messages = case.get("messages", [])
        if not messages and "query" in case:
            messages = [{"role": "user", "content": case["query"]}]
        elif not messages and "prompt" in case:
            messages = [{"role": "user", "content": case["prompt"]}]

        session_id = f"eval_session_{case_id}"
        last_response = None

        for turn in messages:
            if turn.get("role") == "user":
                last_response = agent.process_message(turn.get("content", ""), session_id=session_id)

        # Deterministic Assertions
        passed = True
        reasons = []

        if last_response:
            ans_lower = last_response.answer.lower()

            # 1. Must Include Terms
            expected_terms = case.get("must_include", []) or case.get("expected_terms", [])
            for term in expected_terms:
                if term.lower() not in ans_lower:
                    passed = False
                    reasons.append(f"Missing expected term: '{term}'")

            # 2. Must Exclude Terms (Forbidden)
            forbidden_terms = case.get("must_exclude", []) or case.get("forbidden_terms", [])
            for term in forbidden_terms:
                if term.lower() in ans_lower:
                    passed = False
                    reasons.append(f"Found forbidden term: '{term}'")

            # 3. Source Citation Checks
            req_sources = case.get("required_sources", []) or case.get("expected_citations", [])
            cited_files = [c.filename for c in last_response.citations]
            for src in req_sources:
                if src not in cited_files:
                    passed = False
                    reasons.append(f"Missing required source: '{src}'")

            forb_sources = case.get("forbidden_sources", []) or case.get("forbidden_citations", [])
            for src in forb_sources:
                if src in cited_files:
                    passed = False
                    reasons.append(f"Used superseded/forbidden source: '{src}'")

            # 4. Tool Call Assertions
            expected_tool = case.get("tool") or case.get("expected_tool")
            if expected_tool:
                if expected_tool in ["not_called", "none", None]:
                    if last_response.tool_called is not None:
                        passed = False
                        reasons.append(f"Expected no tool, got '{last_response.tool_called}'")
                else:
                    if last_response.tool_called != expected_tool:
                        passed = False
                        reasons.append(f"Expected tool '{expected_tool}', got '{last_response.tool_called}'")

            # 5. Human Handoff Assertions
            if "handoff" in case or "expected_handoff" in case:
                exp_handoff = case.get("handoff") if "handoff" in case else case.get("expected_handoff")
                if exp_handoff != last_response.human_handoff_recommended:
                    passed = False
                    reasons.append(f"Expected handoff={exp_handoff}, got {last_response.human_handoff_recommended}")

        if passed:
            passed_count += 1
            category_stats[category]["passed"] += 1
            print(f" [PASS] #{idx:02d} [{category.upper()}] {case_id}")
        else:
            print(f" [FAIL] #{idx:02d} [{category.upper()}] {case_id}")
            for r in reasons:
                print(f"        -> {r}")

    # Print Category Scorecard
    print(f"\n=======================================================")
    print(f"                 EVALUATION SUMMARY")
    print(f"=======================================================")
    for cat, stat in category_stats.items():
        pct = (stat["passed"] / stat["total"]) * 100 if stat["total"] > 0 else 0
        print(f"  * {cat.ljust(20)}: {stat['passed']}/{stat['total']} passed ({pct:.1f}%)")
    
    total_pct = (passed_count / len(all_cases)) * 100 if all_cases else 0
    print(f"-------------------------------------------------------")
    print(f"  TOTAL ACCURACY      : {passed_count}/{len(all_cases)} passed ({total_pct:.1f}%)")
    print(f"=======================================================\n")

if __name__ == "__main__":
    run_evaluation_suite()