import pytest

from concordat import Protocol, ProtocolError

REFUND = {
    "start": {"verify_identity": {"verified", "denied"}},
    "verified": {"propose_refund": {"awaiting_approval"}},
    "awaiting_approval": {"human.approve": {"refunded"}, "human.reject": {"denied"}},
}


def test_valid_protocol() -> None:
    p = Protocol("start", REFUND, final={"refunded", "denied"})
    assert p.actions("start") == {"verify_identity"}
    assert p.targets("awaiting_approval", "human.reject") == {"denied"}
    assert p.actions("refunded") == frozenset()


def test_dead_end_state_is_rejected() -> None:
    # "verified" was never given actions, so a session reaching it is stuck.
    with pytest.raises(ProtocolError, match="can never finish.*verified"):
        Protocol("start", {"start": {"verify": {"verified", "done"}}}, final={"done"})


def test_unreachable_state_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="unreachable.*orphan"):
        Protocol(
            "start",
            {"start": {"go": {"done"}}, "orphan": {"go": {"done"}}},
            final={"done"},
        )


def test_loop_without_exit_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="can never finish"):
        Protocol(
            "start",
            {"start": {"a": {"loop"}, "b": {"done"}}, "loop": {"again": {"loop"}}},
            final={"done"},
        )


def test_final_state_with_actions_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="final state 'done' has actions"):
        Protocol("start", {"start": {"go": {"done"}}, "done": {"go": {"done"}}}, final={"done"})


def test_action_without_targets_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="leads nowhere"):
        Protocol("start", {"start": {"go": {"done"}, "noop": set()}}, final={"done"})
