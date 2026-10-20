"""访问控制与文本归一化辅助函数。"""

from __future__ import annotations

import re

DM_POLICIES = {"open", "allowlist", "disabled"}
GROUP_POLICIES = {"open", "allowlist", "disabled"}
INVISIBLE_TEXT_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f]")
MENTION_SPLIT_RE = re.compile(r"[\u2005\s]+")

# \u4f1a\u8bdd\u9694\u79bb\u7528\u7684 session id \u5206\u9694\u7b26\u3002\u5fae\u4fe1\u7684 wxid / \u7fa4\u804a ID \u5747\u4e0d\u4f1a\u5305\u542b\u8be5\u5b57\u7b26\uff0c
# \u56e0\u6b64\u53ef\u4ee5\u65e0\u6b67\u4e49\u5730\u62fc\u63a5\u5e76\u5728\u53d1\u9001\u65f6\u8fd8\u539f\u771f\u5b9e chatId\u3002
UNIQUE_SESSION_SEPARATOR = "%"


def build_isolated_session_id(event) -> str:
    """\u6784\u9020 unique_session \u4f7f\u7528\u7684\u4f1a\u8bdd ID\u3002

    \u7fa4\u804a\u8fd4\u56de ``<sender>%<group>`` \u4ee5\u4fbf\u53d1\u9001\u65f6\u8fd8\u539f\u7fa4\u53f7\uff1b\u79c1\u804a\u8fd4\u56de sender\uff0c
    \u4fdd\u6301\u4e0e\u5f00\u542f\u9694\u79bb\u524d\u7684\u4f1a\u8bdd\u952e\u4e00\u81f4\uff0c\u907f\u514d\u5386\u53f2\u4e0a\u4e0b\u6587\u4e22\u5931\u3002
    """
    sender_id = str(event.get_sender_id() or "")
    group_id = str(event.get_group_id() or "")
    if not group_id:
        return sender_id
    return f"{sender_id}{UNIQUE_SESSION_SEPARATOR}{group_id}"


def resolve_chat_id_from_session(session_id: str | None) -> str:
    """\u4ece\u53ef\u80fd\u88ab\u4f1a\u8bdd\u9694\u79bb\u6539\u5199\u8fc7\u7684 session ID \u8fd8\u539f\u771f\u5b9e\u5fae\u4fe1 chatId\u3002"""
    sid = str(session_id or "")
    if UNIQUE_SESSION_SEPARATOR in sid:
        return sid.rsplit(UNIQUE_SESSION_SEPARATOR, 1)[1]
    return sid


def normalize_wechat_id(raw: str | None) -> str:
    if not raw:
        return ""
    return raw.strip().removeprefix("wechat:").strip()


def normalize_allowlist(values: list[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        normalized = normalize_wechat_id(str(value))
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def strip_leading_mentions(text: str) -> str:
    """移除消息开头的微信 @提及，仅保留真正正文。"""

    cleaned = INVISIBLE_TEXT_RE.sub("", text or "").strip()
    if not cleaned:
        return ""

    parts = [part.strip() for part in MENTION_SPLIT_RE.split(cleaned) if part.strip()]
    while parts and parts[0].startswith(("@", "＠")):
        parts.pop(0)
    if parts:
        return " ".join(parts).strip()
    return cleaned


def _normalize_mention_target(raw: str | None) -> str:
    if not raw:
        return ""
    cleaned = INVISIBLE_TEXT_RE.sub("", str(raw)).strip()
    if cleaned.startswith(("@", "＠")):
        cleaned = cleaned[1:]
    return cleaned.strip().casefold()


def extract_leading_mentions(text: str) -> list[str]:
    """提取消息开头连续 @ 提及目标（已归一化）。"""

    cleaned = INVISIBLE_TEXT_RE.sub("", text or "").strip()
    if not cleaned:
        return []

    parts = [part.strip() for part in MENTION_SPLIT_RE.split(cleaned) if part.strip()]
    mentions: list[str] = []
    for part in parts:
        if not part.startswith(("@", "＠")):
            break
        target = _normalize_mention_target(part)
        if target:
            mentions.append(target)
    return mentions


def is_leading_self_mention(
    text: str, aliases: list[str] | set[str] | tuple[str, ...]
) -> bool:
    """判断消息开头是否 @ 了机器人本人（按别名集合匹配）。"""

    mention_targets = extract_leading_mentions(text)
    if not mention_targets:
        return False

    normalized_aliases = {
        alias
        for alias in (_normalize_mention_target(item) for item in aliases)
        if alias
    }
    if not normalized_aliases:
        return False

    return any(target in normalized_aliases for target in mention_targets)


def is_official_account(chat_id: str) -> bool:
    return normalize_wechat_id(chat_id).startswith("gh_")


def is_group_chat(chat_id: str) -> bool:
    return normalize_wechat_id(chat_id).endswith("@chatroom")


def is_sender_allowed(sender_id: str | None, allowlist: list[str]) -> bool:
    if "*" in allowlist:
        return True
    normalized_sender = normalize_wechat_id(sender_id)
    if not normalized_sender:
        return False
    return normalized_sender in allowlist


def should_forward_message(
    *,
    is_group: bool,
    sender_id: str | None,
    was_mentioned: bool,
    require_mention: bool,
    dm_policy: str,
    dm_allowlist: list[str],
    group_policy: str,
    group_allowlist: list[str],
) -> tuple[bool, str]:
    """返回消息是否应转发到机器人框架。"""

    dm_policy = dm_policy if dm_policy in DM_POLICIES else "disabled"
    group_policy = group_policy if group_policy in GROUP_POLICIES else "disabled"

    if is_group:
        if group_policy == "disabled":
            return False, "group_policy_disabled"
        if group_policy == "allowlist" and not is_sender_allowed(
            sender_id, group_allowlist
        ):
            return False, "group_sender_not_allowlisted"
        if require_mention and not was_mentioned:
            return False, "mention_required"
        return True, f"group_policy_{group_policy}"

    if dm_policy == "disabled":
        return False, "dm_policy_disabled"
    if dm_policy == "allowlist" and not is_sender_allowed(sender_id, dm_allowlist):
        return False, "dm_sender_not_allowlisted"
    return True, f"dm_policy_{dm_policy}"
