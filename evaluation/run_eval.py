import json

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from generation.generate import generate_answer
from retrieval.cli import run_retrieval_pipeline

GOLDEN_SET_PATH = "evaluation/golden_set.json"
REPORT_PATH = "evaluation/report.csv"


def load_golden_set(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_for_eval(query: str) -> dict:
    chunks = run_retrieval_pipeline(query)
    result = generate_answer(query, chunks)
    return {"answer": result.answer, "contexts": [c.text for c in chunks]}


def build_eval_rows(golden_set: list[dict]) -> list[dict]:
    rows = []
    for item in golden_set:
        out = run_pipeline_for_eval(item["query"])
        rows.append(
            {
                "question": item["query"],
                "answer": out["answer"],
                "contexts": out["contexts"],
                "ground_truth": item["ground_truth_answer"],
            }
        )
    return rows


if __name__ == "__main__":
    golden_set = load_golden_set(GOLDEN_SET_PATH)
    rows = build_eval_rows(golden_set)
    report = evaluate(
        Dataset.from_list(rows),
        metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
    )
    report.to_pandas().to_csv(REPORT_PATH)
    print(report)
