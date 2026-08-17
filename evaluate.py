import json

from src.graph import graph


# ============================================================
# LOAD TEST DATA
# ============================================================

with open(
    "data/evaluation/test_questions.json",
    "r",
    encoding="utf-8"
) as file:

    test_questions = json.load(file)


# ============================================================
# METRICS
# ============================================================

total = len(test_questions)

grounded_count = 0
correct_grounded_count = 0
insufficient_context_count = 0
total_retries = 0


results = []


# ============================================================
# RUN EVALUATION
# ============================================================

for index, test in enumerate(test_questions, 1):

    question = test["question"]

    expected_grounded = test.get(
        "expected_grounded",
        True
    )

    print("\n")
    print("=" * 70)
    print(f"TEST {index}/{total}")
    print("=" * 70)

    print(f"QUESTION: {question}")

    try:

        result = graph.invoke(
            {
                "original_question": question,
                "retrieval_query": question,
                "documents": [],
                "answer": "",
                "critique": "",
                "grounded": False,
                "sufficient_context": False,
                "retry_count": 0,
                "trace": []
            }
        )

        grounded = result["grounded"]

        sufficient_context = result[
            "sufficient_context"
        ]

        retries = result[
            "retry_count"
        ]

        total_retries += retries

        # ----------------------------------------------------
        # Groundedness
        # ----------------------------------------------------

        if grounded:
            grounded_count += 1

        # ----------------------------------------------------
        # Expected behavior
        # ----------------------------------------------------

        correct_behavior = (
            grounded == expected_grounded
        )

        if correct_behavior:
            correct_grounded_count += 1

        # ----------------------------------------------------
        # Missing context
        # ----------------------------------------------------

        if not sufficient_context:
            insufficient_context_count += 1

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            {
                "question": question,
                "answer": result["answer"],
                "grounded": grounded,
                "expected_grounded": expected_grounded,
                "sufficient_context": sufficient_context,
                "retry_count": retries,
                "correct_behavior": correct_behavior
            }
        )

        print("\nANSWER:")
        print(result["answer"])

        print(
            f"\nGrounded: {grounded}"
        )

        print(
            f"Context sufficient: {sufficient_context}"
        )

        print(
            f"Retries: {retries}"
        )

        print(
            f"Correct behavior: {correct_behavior}"
        )

    except Exception as error:

        print(
            f"\n❌ TEST FAILED: {error}"
        )


# ============================================================
# FINAL METRICS
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)


if total > 0:

    behavior_accuracy = (
        correct_grounded_count / total
    ) * 100

    grounded_rate = (
        grounded_count / total
    ) * 100

    insufficient_rate = (
        insufficient_context_count / total
    ) * 100

    average_retries = (
        total_retries / total
    )

else:

    behavior_accuracy = 0
    grounded_rate = 0
    insufficient_rate = 0
    average_retries = 0


print(
    f"\nTotal questions: {total}"
)

print(
    f"Grounded answers: {grounded_count}"
)

print(
    f"Grounded rate: {grounded_rate:.2f}%"
)

print(
    f"Correct behavior: {correct_grounded_count}/{total}"
)

print(
    f"Behavior accuracy: {behavior_accuracy:.2f}%"
)

print(
    f"Insufficient-context detections: "
    f"{insufficient_context_count}"
)

print(
    f"Insufficient-context rate: "
    f"{insufficient_rate:.2f}%"
)

print(
    f"Average retries: {average_retries:.2f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "evaluation_results.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "total_questions": total,
            "grounded_answers": grounded_count,
            "grounded_rate": grounded_rate,
            "correct_behavior": correct_grounded_count,
            "behavior_accuracy": behavior_accuracy,
            "insufficient_context_detections":
                insufficient_context_count,
            "insufficient_context_rate":
                insufficient_rate,
            "average_retries":
                average_retries,
            "results": results
        },
        file,
        indent=4
    )


print(
    "\n📊 Results saved to evaluation_results.json"
)