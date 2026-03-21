import json
from pathlib import Path

INPUT_PATH = Path("evaluation/raw_results.json")
OUTPUT_PATH = Path("evaluation/scored_results.json")


def compute_total_score(scores):
    numeric_scores = [
        v for v in scores.values()
        if isinstance(v, (int, float))
    ]
    return sum(numeric_scores)


def compute_status(scores):
    total = compute_total_score(scores)

    # Hard fail
    if scores.get("diagnosis_consistency_score", 2) == 0:
        return "FAIL", total

    if total >= 7:
        return "PASS", total
    elif total >= 4:
        return "REVISE", total
    else:
        return "FAIL", total


def extract_scores(pipeline_output):
    # Assumes critic_scores exist in output
    return pipeline_output.get("critic_scores", {})


def extract_final_status(pipeline_output):
    return pipeline_output.get("final_status", "UNKNOWN")


def main():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)["results"]

    scored = []

    for item in data:
        output = item.get("pipeline_output", {})

        scores = extract_scores(output)

        if not scores:
            scored.append({
                "case_id": item["case_id"],
                "scenario_type": item.get("scenario_type"),
                "expected_behavior": item.get("expected_behavior"),
                "actual_final_status": extract_final_status(output),
                "computed_final_status": "FAIL",
                "total_score": 0,
                "critic_scores": {},
                "error": "missing_critic_scores"
            })
            continue

        computed_status, total_score = compute_status(scores)

        scored.append({
            "case_id": item["case_id"],
            "scenario_type": item["scenario_type"],
            "expected_behavior": item["expected_behavior"],
            "actual_final_status": extract_final_status(output),
            "computed_final_status": computed_status,
            "total_score": total_score,
            "critic_scores": scores
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"results": scored}, f, indent=2)


if __name__ == "__main__":
    main()
