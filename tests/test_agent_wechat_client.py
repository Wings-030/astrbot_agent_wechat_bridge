from src.agent_wechat_client import WeChatClient


def test_build_events_ws_url_with_token() -> None:
    client = WeChatClient("http://localhost:6174", token="secret")
    assert client.build_events_ws_url() == "ws://localhost:6174/api/ws/events?token=secret"


def test_build_events_ws_url_https() -> None:
    client = WeChatClient("https://example.com/base", token=None)
    assert client.build_events_ws_url() == "wss://example.com/base/api/ws/events"


def test_send_message_supports_timeout_override(monkeypatch) -> None:
    client = WeChatClient("http://localhost:6174", token="secret")
    captured: dict[str, object] = {}

    def fake_post(path, body=None, *, timeout=None):
        captured["path"] = path
        captured["body"] = body
        captured["timeout"] = timeout
        return {"success": True}

    monkeypatch.setattr(client, "_post", fake_post)
    payload = {"chatId": "wxid_x", "text": "hello"}
    result = client.send_message(payload, timeout=30.0)

    assert result == {"success": True}
    assert captured["path"] == "/api/messages/send"
    assert captured["body"] == payload
    assert captured["timeout"] == 30.0


def test_send_voice_posts_multipart_with_idempotency_key(monkeypatch) -> None:
    client = WeChatClient("http://localhost:6174", token="secret")
    captured: dict[str, object] = {}

    class FakeResponse:
        ok = True
        text = ""

        def json(self):
            return {"jobId": "job-1", "status": "queued"}

    class FakeSession:
        def post(self, url, data=None, files=None, headers=None, timeout=None):
            captured["url"] = url
            captured["data"] = data
            captured["files"] = files
            captured["headers"] = headers
            captured["timeout"] = timeout
            return FakeResponse()

    monkeypatch.setattr(client, "_session", lambda: FakeSession())

    result = client.send_voice(
        "wxid_x",
        b"audio-bytes",
        idempotency_key="key-1",
        filename="tts.mp3",
        mime_type="audio/mpeg",
        timeout=30.0,
    )

    assert result == {"jobId": "job-1", "status": "queued"}
    assert captured["url"] == "http://localhost:6174/api/messages/voice"
    assert captured["data"] == {"chatId": "wxid_x"}
    assert captured["files"] == {"audio": ("tts.mp3", b"audio-bytes", "audio/mpeg")}
    assert captured["headers"] == {"Content-Type": None, "Idempotency-Key": "key-1"}
    assert captured["timeout"] == 30.0


def test_send_voice_requires_idempotency_key() -> None:
    client = WeChatClient("http://localhost:6174")

    try:
        client.send_voice("wxid_x", b"audio", idempotency_key="")
    except ValueError as exc:
        assert "idempotency_key" in str(exc)
    else:
        raise AssertionError("expected ValueError")
