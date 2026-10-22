import pytest

from concordat import Protocol, ProtocolViolation, Session

REFUND = Protocol(
    "start",
    {
        "start": {"verify_identity": {"verified", "denied"}},
        "verified": {"propose_refund": {"awaiting_approval"}},
        "awaiting_approval": {"human.approve": {"refunded"}, "human.reject": {"denied"}},
    },
    final={"refunded", "denied"},
)


def test_happy_path() -> None:
    s = Session(REFUND)
    s.step("verify_identity", "verified")
    s.step("propose_refund")  # single target, so it can be omitted
    assert s.allowed_actions() == {"human.approve", "human.reject"}
    s.step("human.approve")
    assert s.done
    assert s.state == "refunded"
    assert [a for _, a, _ in s.history] == ["verify_identity", "propose_refund", "human.approve"]


def test_refund_before_verification_is_rejected() -> None:
    s = Session(REFUND)
    with pytest.raises(ProtocolViolation, match="'propose_refund' is not allowed in state 'start'"):
        s.step("propose_refund")
    assert s.state == "start"


def test_illegal_target_is_rejected() -> None:
    s = Session(REFUND)
    with pytest.raises(ProtocolViolation, match="can lead to"):
        s.step("verify_identity", "refunded")


def test_ambiguous_target_must_be_given() -> None:
    s = Session(REFUND)
    with pytest.raises(ProtocolViolation, match="not None"):
        s.step("verify_identity")
