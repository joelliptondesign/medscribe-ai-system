import json
from pathlib import Path

INPUT_PATH = Path("evaluation/scored_results.json")
OUTPUT_PATH = Path("evaluation/eval_summary.json")


def main():
    with open(INPUT_PATH, "r") as f:
        results = json.load(f)["results"]

    total = len(results)

    pass_count = 0
    revise_count = 0
    fail_count = 0

    correct_behavior = 0
    status_changed = 0
    unchanged = 0

    corrections_to_fail = 0
    corrections_to_revise = 0

    total_score_sum = 0

    for r in results:
        expected = r.get("expected_behavior")
        actual = r.get("actual_final_status")
        computed = r.get("computed_final_status")
        score = r.get("total_score", 0)

        total_score_sum += score

        # Distribution
        if computed == "PASS":
            pass_count += 1
        elif computed == "REVISE":
            revise_count += 1
        elif computed == "FAIL":
            fail_count += 1

        # Behavior accuracy
        if computed == expected:
            correct_behavior += 1

        # Comparison
        if actual != computed:
            status_changed += 1

            if computed == "FAIL":
                corrections_to_fail += 1
            elif computed == "REVISE":
                corrections_to_revise += 1
        else:
            unchanged += 1

    summary = {
        "total_cases": total,
        "behavior_accuracy": correct_behavior / total,
        "distribution": {
            "pass_rate": pass_count / total,
            "revise_rate": revise_count / total,
            "fail_rate": fail_count / total
        },
        "governance_impact": {
            "status_change_rate": status_changed / total,
            "unchanged_rate": unchanged / total,
            "corrections_to_fail": corrections_to_fail,
            "corrections_to_revise": corrections_to_revise
        },
        "average_score": total_score_sum / total
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
