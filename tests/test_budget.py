import pytest

from concordat import Budget, BudgetExceeded


def test_spend_deducts() -> None:
    b = Budget(tokens=100, tool_calls=2)
    b.spend(tokens=40, tool_calls=1)
    assert (b.tokens, b.tool_calls, b.usd) == (60, 1, None)


def test_overspend_raises_and_deducts_nothing() -> None:
    b = Budget(tokens=100, tool_calls=1)
    with pytest.raises(BudgetExceeded, match="tool_calls"):
        b.spend(tokens=10, tool_calls=2)
    assert (b.tokens, b.tool_calls) == (100, 1)


def test_unlimited_kinds_are_not_tracked() -> None:
    b = Budget(usd=1.0)
    b.spend(tokens=10**9, usd=0.25)
    assert (b.tokens, b.usd) == (None, 0.75)


def test_split_moves_allowance_to_child() -> None:
    parent = Budget(tokens=1000, tool_calls=10)
    child = parent.split(tokens=300, tool_calls=4)
    assert (child.tokens, child.tool_calls) == (300, 4)
    assert (parent.tokens, parent.tool_calls) == (700, 6)


def test_split_cannot_exceed_parent() -> None:
    parent = Budget(tokens=100, tool_calls=1)
    with pytest.raises(BudgetExceeded):
        parent.split(tokens=200, tool_calls=1)
    assert parent.tokens == 100


def test_child_cannot_be_unlimited_where_parent_is_limited() -> None:
    with pytest.raises(BudgetExceeded, match="can't be unlimited"):
        Budget(tokens=100).split()


# Reported by Halil Talha Ertan: negative amounts created allowance, and USD never stopped a run.


@pytest.mark.parametrize("amount", [-1, float("nan"), float("inf")])
def test_spend_rejects_negative_and_non_finite_amounts(amount: float) -> None:
    b = Budget(tokens=10, usd=1.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        b.spend(tokens=amount)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite and non-negative"):
        Budget().spend(usd=amount)  # unlimited dimensions are checked too
    assert (b.tokens, b.usd) == (10, 1.0)


def test_negative_split_cannot_create_allowance() -> None:
    parent = Budget(tokens=100, tool_calls=10, usd=1.0)
    with pytest.raises(ValueError):
        parent.split(tokens=-20, tool_calls=-3, usd=-0.5)
    assert (parent.tokens, parent.tool_calls, parent.usd) == (100, 10, 1.0)


@pytest.mark.parametrize(
    "kwargs", [{"tokens": -5}, {"usd": float("nan")}, {"tool_calls": float("inf")}]
)
def test_construction_rejects_invalid_amounts(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Budget(**kwargs)  # type: ignore[arg-type]


def test_exhausted_covers_every_limited_amount() -> None:
    assert Budget(usd=0.0).exhausted()
    assert Budget(tool_calls=0).exhausted()
    assert not Budget(tokens=1, usd=0.5).exhausted()
    assert not Budget().exhausted()
