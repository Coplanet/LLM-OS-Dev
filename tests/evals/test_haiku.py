from typing import Optional

from phi.eval import Eval, EvalResult

from ai.leaders.generic import get_leader


def test_haiku():
    evaluation = Eval(
        agent=get_leader(),
        question="Share a haiku",
        expected_answer="Any haiku",
    )
    result: Optional[EvalResult] = evaluation.print_result()

    assert result is not None and result.accuracy_score >= 7


test_haiku()
