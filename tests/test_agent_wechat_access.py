from src.agent_wechat_access import (
    build_isolated_session_id,
    extract_leading_mentions,
    is_leading_self_mention,
    is_group_chat,
    is_sender_allowed,
    normalize_allowlist,
    normalize_wechat_id,
    resolve_chat_id_from_session,
    should_forward_message,
    strip_leading_mentions,
)


def test_normalize_wechat_id() -> None:
    assert normalize_wechat_id("wechat:wxid_abc") == "wxid_abc"
    assert normalize_wechat_id("  wxid_abc  ") == "wxid_abc"


def test_normalize_allowlist_deduplicates() -> None:
    assert normalize_allowlist(["wxid_a", "wechat:wxid_a", "wxid_b"]) == [
        "wxid_a",
        "wxid_b",
    ]


def test_strip_leading_mentions() -> None:
    text = "@Bot\u2005@Alice\u2005 hello there"
    assert strip_leading_mentions(text) == "hello there"


def test_extract_leading_mentions() -> None:
    text = "@小暹罗\u2005@Alice\u2005 hello there"
    assert extract_leading_mentions(text) == ["小暹罗", "alice"]


def test_is_leading_self_mention() -> None:
    text = "@小暹罗\u2005hi"
    assert is_leading_self_mention(text, {"wx-小暹罗", "小暹罗"}) is True
    assert is_leading_self_mention("@Chill\u2005hi", {"小暹罗"}) is False


def test_group_chat_detection() -> None:
    assert is_group_chat("123@chatroom") is True
    assert is_group_chat("wxid_123") is False


def test_sender_allowlist() -> None:
    assert is_sender_allowed("wxid_abc", ["wxid_abc"]) is True
    assert is_sender_allowed("wxid_other", ["wxid_abc"]) is False
    assert is_sender_allowed("wxid_any", ["*"]) is True


def test_should_forward_direct_message() -> None:
    allowed, reason = should_forward_message(
        is_group=False,
        sender_id="wxid_abc",
        was_mentioned=False,
        require_mention=True,
        dm_policy="allowlist",
        dm_allowlist=["wxid_abc"],
        group_policy="disabled",
        group_allowlist=[],
    )
    assert allowed is True
    assert reason == "dm_policy_allowlist"


def test_should_block_group_without_mention() -> None:
    allowed, reason = should_forward_message(
        is_group=True,
        sender_id="wxid_abc",
        was_mentioned=False,
        require_mention=True,
        dm_policy="open",
        dm_allowlist=[],
        group_policy="open",
        group_allowlist=[],
    )
    assert allowed is False
    assert reason == "mention_required"


class _FakeEvent:
    def __init__(self, sender_id: str, group_id: str | None = None) -> None:
        self._sender_id = sender_id
        self._group_id = group_id

    def get_sender_id(self) -> str:
        return self._sender_id

    def get_group_id(self) -> str | None:
        return self._group_id


def test_build_isolated_session_id_roundtrips_group() -> None:
    sid = build_isolated_session_id(_FakeEvent("wxid_a_b", "123456@chatroom"))
    assert sid == "wxid_a_b%123456@chatroom"
    assert resolve_chat_id_from_session(sid) == "123456@chatroom"


def test_build_isolated_session_id_keeps_private_sender() -> None:
    assert build_isolated_session_id(_FakeEvent("wxid_a_b", None)) == "wxid_a_b"


def test_resolve_chat_id_passes_through_plain_session() -> None:
    assert resolve_chat_id_from_session("123456@chatroom") == "123456@chatroom"
