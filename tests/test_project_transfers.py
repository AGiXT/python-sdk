"""Offline fork compatibility tests. Never import/run the live SDK test suite."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock
import json
import traceback
import uuid

import pytest
import requests
from requests.api import get as http_get, post as http_post

import agixtsdk
from agixtsdk import AGiXTSDK, ProjectTransferError


SOURCE, MESSAGE, REQUEST, AGENT, COMPANY, TARGET, REPOSITORY = [
    str(uuid.uuid4()) for _ in range(7)
]
DIGEST = "a" * 64
TOKEN = "private-test-credential"


def receipt(state="queued", *, company=COMPANY):
    result = {
        "transfer": {
            "request_id": REQUEST,
            "state": state,
            "error": None,
            "source_conversation_id": SOURCE,
            "conversation_id": TARGET if state == "completed" else None,
            "agent_id": AGENT,
            "company_id": company,
            "summary": None,
        }
    }
    if state == "completed":
        result.update(id=TARGET, name="Original (fork)")
    if state in ("attention", "failed"):
        result["transfer"]["error"] = "history_storage_limit"
    return result


def response(data, status=200):
    return Mock(status_code=status, json=Mock(return_value=data))


@pytest.fixture(autouse=True)
def http(monkeypatch):
    post = Mock(side_effect=AssertionError("Unexpected POST"))
    get = Mock(side_effect=AssertionError("Unexpected GET"))
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", get)
    return post, get


@pytest.fixture
def sdk():
    return AGiXTSDK(base_uri="https://sdk.invalid", api_key=TOKEN)


def serve(mock, *data):
    mock.side_effect = [response(item) for item in data]


def test_owning_checkout():
    assert Path(agixtsdk.__file__).resolve() == (
        Path(__file__).resolve().parents[1] / "agixtsdk" / "__init__.py"
    )


@pytest.mark.parametrize("legacy", [{"id": TARGET, "name": "Fork"}, "Fork", {}])
@pytest.mark.parametrize("source", [SOURCE, "-"])
def test_legacy_positional_signature_and_json_unchanged(sdk, http, legacy, source):
    post, get = http
    serve(post, legacy)
    original_headers = sdk.headers.copy()
    assert sdk.fork_conversation(source, MESSAGE) is legacy
    args = post.call_args.kwargs
    assert args["url"] == (
        f"https://sdk.invalid/v1/conversation/fork/{source}/{MESSAGE}"
    )
    assert set(args) == {"url", "headers", "timeout", "allow_redirects"}
    assert args["allow_redirects"] is False
    assert uuid.UUID(args["headers"]["X-Request-ID"]).version == 4
    assert args["headers"]["Authorization"] == TOKEN
    assert sdk.headers == original_headers
    post.assert_called_once()
    get.assert_not_called()


def test_alias_native_receipt_pins_resolved_source(sdk, http, monkeypatch):
    pending, running, done = receipt(), receipt("running"), receipt("completed")
    serve(http[0], pending)
    serve(http[1], running, done)
    observe = Mock(wraps=sdk._get_project_transfer)
    monkeypatch.setattr(sdk, "_get_project_transfer", observe)
    monkeypatch.setattr(agixtsdk.time, "sleep", Mock())

    assert (
        sdk.fork_conversation(
            "-", MESSAGE, request_id=REQUEST, agent_id=AGENT, company_id=COMPANY
        )
        is done
    )
    assert http[0].call_args.kwargs["url"].endswith(f"/fork/-/{MESSAGE}")
    http[0].assert_called_once()
    assert observe.call_count == http[1].call_count == 2
    for call in observe.call_args_list:
        assert call.args[:4] == (REQUEST, AGENT, COMPANY, SOURCE)
    for call in http[1].call_args_list:
        assert call.kwargs["url"].endswith(f"/project-transfers/{REQUEST}")
        assert call.kwargs["params"] == {"agent_id": AGENT, "company_id": COMPANY}


def test_alias_native_wait_false_returns_original_receipt(sdk, http):
    pending = receipt()
    serve(http[0], pending)
    assert (
        sdk.fork_conversation("-", MESSAGE, request_id=REQUEST, wait=False) is pending
    )
    http[0].assert_called_once()
    http[1].assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", TARGET),
        ("agent_id", TARGET),
        ("company_id", TARGET),
        ("company_id", None),
        ("source_conversation_id", None),
        ("source_conversation_id", "-"),
        ("source_conversation_id", str(uuid.UUID(int=0))),
        ("source_conversation_id", SOURCE.upper()),
        ("state", "unknown"),
    ],
)
def test_alias_first_receipt_still_validates_identity(sdk, http, field, value):
    invalid = receipt()
    invalid["transfer"][field] = value
    serve(http[0], invalid)
    with pytest.raises(ProjectTransferError):
        sdk.fork_conversation(
            "-", MESSAGE, request_id=REQUEST, agent_id=AGENT, company_id=COMPANY
        )
    http[0].assert_called_once()
    http[1].assert_not_called()


@pytest.mark.parametrize("state", ["running", "completed"])
def test_alias_poll_rejects_changed_source(sdk, http, state):
    serve(http[0], receipt())
    invalid = receipt(state)
    invalid["transfer"]["source_conversation_id"] = TARGET
    serve(http[1], invalid)
    with pytest.raises(ProjectTransferError):
        sdk.fork_conversation("-", MESSAGE, request_id=REQUEST)
    http[0].assert_called_once()
    http[1].assert_called_once()


@pytest.mark.parametrize("source", [" -", "- ", "other-name"])
def test_only_exact_alias_accepts_server_resolved_native_source(sdk, http, source):
    serve(http[0], receipt())
    with pytest.raises(ProjectTransferError):
        sdk.fork_conversation(source, MESSAGE, request_id=REQUEST)
    http[0].assert_called_once()
    http[1].assert_not_called()


def test_selected_alias_refused_before_post(sdk, http):
    with pytest.raises(ValueError, match="source conversation UUID, not '-'"):
        sdk.fork_conversation("-", MESSAGE, preview_digest=DIGEST, repositories=[])
    for method in http:
        method.assert_not_called()


@pytest.mark.parametrize("repositories", [[], [REPOSITORY]])
def test_explicit_selection_preserved(sdk, http, repositories):
    post, get = http
    data = receipt("completed")
    serve(post, data)
    assert (
        sdk.fork_conversation(
            SOURCE,
            MESSAGE,
            request_id=REQUEST,
            agent_id=AGENT,
            company_id=COMPANY,
            preview_digest=DIGEST,
            repositories=repositories,
        )
        is data
    )
    args = post.call_args.kwargs
    assert args["url"] == (
        f"https://sdk.invalid/v1/computer/project-transfers/fork/{SOURCE}/{MESSAGE}"
    )
    assert args["json"] == {"preview_digest": DIGEST, "repositories": repositories}
    assert args["headers"]["X-Request-ID"] == REQUEST
    assert args["headers"]["X-Agent-ID"] == AGENT
    assert args["headers"]["X-Company-ID"] == COMPANY
    assert args["headers"] is not sdk.headers
    post.assert_called_once()
    get.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"preview_digest": DIGEST},
        {"repositories": []},
        {"preview_digest": "", "repositories": []},
        {"preview_digest": DIGEST, "repositories": "all"},
        {"preview_digest": DIGEST, "repositories": [REPOSITORY, REPOSITORY]},
        {"request_id": "not-a-uuid"},
        {"request_id": str(uuid.UUID(int=0))},
        {"timeout": -1},
        {"timeout": float("inf")},
        {"request_timeout": 0},
        {"poll_interval": 0},
    ],
)
def test_invalid_options_never_post(sdk, http, options):
    with pytest.raises(ValueError):
        sdk.fork_conversation(SOURCE, MESSAGE, **options)
    for method in http:
        method.assert_not_called()


def test_selection_refusal_does_not_fallback_or_select_all(sdk, http):
    post, get = http
    error = {"detail": "repository_selection_changed"}
    post.side_effect = [response(error, 409)]
    assert (
        sdk.fork_conversation(
            SOURCE, MESSAGE, preview_digest=DIGEST, repositories=[REPOSITORY]
        )
        is error
    )
    post.assert_called_once()
    get.assert_not_called()


def test_unselected_repository_refusal_is_returned_without_any_fallback(sdk, http):
    error = {"detail": "repository_selection_required"}
    http[0].side_effect = [response(error, 409)]
    assert sdk.fork_conversation(SOURCE, MESSAGE) is error
    assert "json" not in http[0].call_args.kwargs
    http[0].assert_called_once()
    http[1].assert_not_called()


def test_selected_success_requires_transfer_identity(sdk, http):
    serve(http[0], {"id": TARGET, "name": "Unexpected legacy-only response"})
    with pytest.raises(ProjectTransferError):
        sdk.fork_conversation(
            SOURCE, MESSAGE, preview_digest=DIGEST, repositories=[REPOSITORY]
        )
    http[0].assert_called_once()
    http[1].assert_not_called()


def test_preview_is_read_only(sdk, http):
    post, get = http
    data = {
        "source_conversation_id": SOURCE,
        "preview_digest": DIGEST,
        "repositories": [{"checkout_id": REPOSITORY}],
    }
    serve(get, data)
    original = sdk.headers.copy()
    assert sdk.preview_conversation_fork(SOURCE, company_id=COMPANY) is data
    assert get.call_args.kwargs["url"] == (
        f"https://sdk.invalid/v1/computer/project-transfers/fork/{SOURCE}"
    )
    assert "params" not in get.call_args.kwargs
    assert get.call_args.kwargs["headers"]["X-Company-ID"] == COMPANY
    assert sdk.headers == original
    post.assert_not_called()


def test_preview_rejects_wrong_source(sdk, http):
    serve(http[1], {"source_conversation_id": TARGET, "preview_digest": DIGEST})
    with pytest.raises(ProjectTransferError, match="source mismatch"):
        sdk.preview_conversation_fork(SOURCE)


def test_default_polls_with_original_credentials_and_resolved_scope(
    sdk, http, monkeypatch
):
    post, get = http
    pending, running, done = receipt(), receipt("running"), receipt("completed")

    def submit(**kwargs):
        sdk.headers = {"Authorization": "replacement-token", "X-Company-ID": TARGET}
        return response(pending, 202)

    post.side_effect = submit
    serve(get, running, done)
    sleep = Mock()
    monkeypatch.setattr(agixtsdk.time, "sleep", sleep)
    assert sdk.fork_conversation(SOURCE, MESSAGE, request_id=REQUEST) is done
    assert "json" not in post.call_args.kwargs
    post.assert_called_once()
    assert get.call_count == 2
    for call in get.call_args_list:
        args = call.kwargs
        assert args["url"] == (
            f"https://sdk.invalid/v1/computer/project-transfers/{REQUEST}"
        )
        assert args["params"] == {"agent_id": AGENT, "company_id": COMPANY}
        assert args["headers"]["Authorization"] == TOKEN
        assert args["headers"]["X-Request-ID"] == REQUEST
        assert args["headers"] is post.call_args.kwargs["headers"]
    sleep.assert_called_once_with(2.0)


def test_verbose_handles_accepted_response_without_assert_or_credentials(
    sdk, http, capsys
):
    sdk.verbose = True
    http[0].side_effect = [response(receipt(), 202)]
    done = receipt("completed")
    serve(http[1], done)
    assert sdk.fork_conversation(SOURCE, MESSAGE, request_id=REQUEST) is done
    assert TOKEN not in capsys.readouterr().out


def test_polling_pins_null_company_scope(sdk, http):
    serve(http[0], receipt(company=None))
    serve(http[1], receipt("completed", company=COMPANY))
    with pytest.raises(ProjectTransferError):
        sdk.fork_conversation(SOURCE, MESSAGE, request_id=REQUEST)
    assert http[1].call_args.kwargs["params"] == {"agent_id": AGENT}
    http[0].assert_called_once()


@pytest.mark.parametrize("state", ["completed", "failed", "attention"])
def test_terminal_post_response_not_polled(sdk, http, state):
    data = receipt(state)
    serve(http[0], data)
    assert sdk.fork_conversation(SOURCE, MESSAGE, request_id=REQUEST) is data
    http[1].assert_not_called()


@pytest.mark.parametrize("state", ["failed", "attention"])
def test_polled_refusal_returned_without_resume(sdk, http, state):
    pending, terminal = receipt(), receipt(state)
    serve(http[0], pending)
    serve(http[1], terminal)
    assert sdk.fork_conversation(SOURCE, MESSAGE, request_id=REQUEST) is terminal
    http[0].assert_called_once()
    http[1].assert_called_once()


@pytest.mark.parametrize("options", [{"wait": False}, {"timeout": 0}])
def test_no_wait_returns_original_receipt(sdk, http, options):
    data = receipt()
    serve(http[0], data)
    assert sdk.fork_conversation(SOURCE, MESSAGE, request_id=REQUEST, **options) is data
    http[1].assert_not_called()


def test_exhausted_wait_returns_original_not_latest_receipt(sdk, http, monkeypatch):
    pending, running = receipt(), receipt("running")
    serve(http[0], pending)
    serve(http[1], running)
    monkeypatch.setattr(agixtsdk.time, "monotonic", Mock(side_effect=[0, 0, 1, 2]))
    sleep = Mock()
    monkeypatch.setattr(agixtsdk.time, "sleep", sleep)
    assert (
        sdk.fork_conversation(
            SOURCE, MESSAGE, request_id=REQUEST, timeout=2, request_timeout=30
        )
        is pending
    )
    assert http[1].call_args.kwargs["timeout"] == 2
    sleep.assert_called_once_with(1)
    http[0].assert_called_once()
    http[1].assert_called_once()


def test_budget_exhausted_during_get_timeout_returns_original(sdk, http, monkeypatch):
    pending = receipt()
    serve(http[0], pending)
    http[1].side_effect = requests.Timeout(f"Credential: {TOKEN}")
    monkeypatch.setattr(agixtsdk.time, "monotonic", Mock(side_effect=[0, 0, 2]))
    assert (
        sdk.fork_conversation(SOURCE, MESSAGE, request_id=REQUEST, timeout=2) is pending
    )
    assert http[1].call_args.kwargs["timeout"] == 2
    http[0].assert_called_once()
    http[1].assert_called_once()


def test_context_mismatch_not_suppressed_at_deadline(sdk, http, monkeypatch):
    serve(http[0], receipt())
    invalid = receipt("completed")
    invalid["transfer"]["agent_id"] = TARGET
    serve(http[1], invalid)
    monkeypatch.setattr(agixtsdk.time, "monotonic", Mock(side_effect=[0, 0, 2]))
    with pytest.raises(ProjectTransferError):
        sdk.fork_conversation(SOURCE, MESSAGE, request_id=REQUEST, timeout=2)


@pytest.mark.parametrize("company", [None, COMPANY])
def test_status_exact_query_and_headers(sdk, http, company):
    data = receipt("completed", company=company)
    serve(http[1], data)
    sdk.headers["x-company-id"] = TARGET
    original = sdk.headers.copy()
    assert (
        sdk.get_project_transfer(REQUEST, AGENT, company, source_conversation_id=SOURCE)
        is data
    )
    params = {"agent_id": AGENT}
    if company:
        params["company_id"] = company
    args = http[1].call_args.kwargs
    assert args["params"] == params
    assert args["url"].endswith(f"/project-transfers/{REQUEST}")
    assert args["headers"].get("X-Company-ID") == company
    assert args["headers"]["X-Agent-ID"] == AGENT
    assert args["headers"]["X-Request-ID"] == REQUEST
    assert sdk.headers == original
    http[0].assert_not_called()


@pytest.mark.parametrize("stage", ["post", "poll", "status"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", TARGET),
        ("agent_id", TARGET),
        ("company_id", TARGET),
        ("company_id", None),
        ("source_conversation_id", TARGET),
        ("source_conversation_id", None),
        ("state", "unknown"),
    ],
)
def test_context_mismatch_rejected(sdk, http, stage, field, value):
    bad = receipt("completed")
    bad["transfer"][field] = value
    if stage == "post":
        serve(http[0], bad)
    else:
        serve(http[0], receipt())
        serve(http[1], bad)
    with pytest.raises(ProjectTransferError) as error:
        if stage == "status":
            sdk.get_project_transfer(
                REQUEST, AGENT, COMPANY, source_conversation_id=SOURCE
            )
        else:
            sdk.fork_conversation(
                SOURCE, MESSAGE, request_id=REQUEST, agent_id=AGENT, company_id=COMPANY
            )
    assert error.value.request_id == REQUEST
    assert http[0].call_count == (0 if stage == "status" else 1)


@pytest.mark.parametrize("missing", ["request_id", "agent_id", "company_id", "state"])
def test_missing_identity_rejected(sdk, http, missing):
    data = receipt()
    del data["transfer"][missing]
    serve(http[0], data)
    with pytest.raises(ProjectTransferError):
        sdk.fork_conversation(SOURCE, MESSAGE, request_id=REQUEST)
    http[1].assert_not_called()


@pytest.mark.parametrize("change", [{"id": SOURCE}, {"name": None}])
def test_completed_response_must_have_consistent_legacy_result(sdk, http, change):
    data = receipt("completed")
    data.update(change)
    serve(http[0], data)
    with pytest.raises(ProjectTransferError):
        sdk.fork_conversation(SOURCE, MESSAGE, request_id=REQUEST)


def test_successful_status_requires_transfer_identity(sdk, http):
    serve(http[1], {"id": TARGET, "name": "Not a transfer receipt"})
    with pytest.raises(ProjectTransferError):
        sdk.get_project_transfer(REQUEST, AGENT, COMPANY)


def test_status_access_refusal_is_not_reauthorized(sdk, http):
    error = {"detail": "Current recipient access denied"}
    http[1].side_effect = [response(error, 403)]
    assert sdk.get_project_transfer(REQUEST, AGENT, COMPANY) is error
    http[0].assert_not_called()
    http[1].assert_called_once()


@pytest.mark.parametrize("status", [300, 301, 302, 303, 304, 307, 308, 399])
@pytest.mark.parametrize(
    "operation", ["post", "selected_post", "poll", "status", "preview"]
)
def test_transport_redirect_never_followed_or_exposed(
    sdk, monkeypatch, capsys, operation, status
):
    # Keep real Requests redirect handling, but intercept every transport send.
    monkeypatch.setattr(requests, "post", http_post)
    monkeypatch.setattr(requests, "get", http_get)
    sdk.verbose = True
    sent = []
    redirect = requests.Response()
    redirect.status_code = status
    redirect.headers["Location"] = f"https://redirect.invalid/{TOKEN}"
    redirect._content = f"Redirect containing {TOKEN}".encode()
    redirect.json = Mock(side_effect=AssertionError("Redirect body must not be parsed"))

    def send(adapter, request, **kwargs):
        sent.append(request)
        if operation == "poll" and len(sent) == 1:
            result = requests.Response()
            result.status_code = 202
            result._content = json.dumps(receipt()).encode()
        else:
            assert len(sent) == (2 if operation == "poll" else 1)
            result = redirect
        result.url = request.url
        result.request = request
        return result

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    with pytest.raises(ProjectTransferError, match="redirect refused") as error:
        if operation == "preview":
            sdk.preview_conversation_fork(SOURCE)
        elif operation == "status":
            sdk.get_project_transfer(REQUEST, AGENT, COMPANY)
        else:
            options = {"request_id": REQUEST}
            if operation == "selected_post":
                options.update(preview_digest=DIGEST, repositories=[])
            sdk.fork_conversation(SOURCE, MESSAGE, **options)

    assert [request.method for request in sent] == (
        ["POST", "GET"]
        if operation == "poll"
        else ["GET" if operation in ("status", "preview") else "POST"]
    )
    assert all(request.url.startswith("https://sdk.invalid/") for request in sent)
    assert error.value.request_id == (None if operation == "preview" else REQUEST)
    assert not error.value.timed_out
    rendered = "".join(traceback.format_exception(error.value))
    output = capsys.readouterr()
    assert TOKEN not in rendered + output.out + output.err
    assert "redirect.invalid" not in rendered + output.out + output.err
    redirect.json.assert_not_called()


@pytest.mark.parametrize("operation", ["post", "status", "preview"])
def test_invalid_json_exposes_only_safe_request_id(sdk, http, operation):
    invalid = Mock(status_code=200)
    invalid.json.side_effect = ValueError(f"Invalid JSON containing {TOKEN}")
    serve(http[0], receipt())
    if operation == "post":
        http[0].side_effect = [invalid]
    else:
        http[1].side_effect = [invalid]
    with pytest.raises(ProjectTransferError) as error:
        if operation == "preview":
            sdk.preview_conversation_fork(SOURCE)
        elif operation == "status":
            sdk.get_project_transfer(REQUEST, AGENT, COMPANY)
        else:
            sdk.fork_conversation(SOURCE, MESSAGE, request_id=REQUEST)
    assert error.value.request_id == (None if operation == "preview" else REQUEST)
    assert TOKEN not in "".join(traceback.format_exception(error.value))
    assert http[0].call_count == (1 if operation == "post" else 0)


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("known_request", [False, True])
def test_post_network_loss_exposes_safe_request_id_never_retries(
    sdk, http, selected, known_request
):
    http[0].side_effect = requests.ConnectionError(
        f"Disconnected Authorization: {TOKEN}"
    )
    options = {"request_id": REQUEST} if known_request else {}
    if selected:
        options.update(preview_digest=DIGEST, repositories=[REPOSITORY])
    with pytest.raises(ProjectTransferError) as error:
        sdk.fork_conversation(SOURCE, MESSAGE, **options)
    request_id = http[0].call_args.kwargs["headers"]["X-Request-ID"]
    assert error.value.request_id == request_id
    assert request_id in str(error.value)
    assert TOKEN not in "".join(traceback.format_exception(error.value))
    assert uuid.UUID(request_id)
    http[0].assert_called_once()
    http[1].assert_not_called()


@pytest.mark.parametrize("operation", ["poll", "status", "preview"])
def test_get_network_loss_never_posts_again(sdk, http, operation):
    serve(http[0], receipt())
    http[1].side_effect = requests.Timeout(f"Authorization: {TOKEN}")
    with pytest.raises(ProjectTransferError) as error:
        if operation == "preview":
            sdk.preview_conversation_fork(SOURCE)
        elif operation == "status":
            sdk.get_project_transfer(REQUEST, AGENT, COMPANY)
        else:
            sdk.fork_conversation(SOURCE, MESSAGE, request_id=REQUEST)
    assert error.value.request_id == (None if operation == "preview" else REQUEST)
    assert TOKEN not in "".join(traceback.format_exception(error.value))
    assert http[0].call_count == (1 if operation == "poll" else 0)


def test_per_call_scope_overrides_case_insensitive_headers(sdk, http):
    sdk.headers.update(
        {"x-request-id": TARGET, "x-agent-id": TARGET, "x-company-id": TARGET}
    )
    original = sdk.headers.copy()
    serve(http[0], receipt("completed"))
    sdk.fork_conversation(
        SOURCE, MESSAGE, request_id=REQUEST, agent_id=AGENT, company_id=COMPANY
    )
    headers = http[0].call_args.kwargs["headers"]
    assert len([key for key in headers if key.lower() == "x-request-id"]) == 1
    assert headers["X-Request-ID"] == REQUEST
    assert headers["X-Agent-ID"] == AGENT
    assert headers["X-Company-ID"] == COMPANY
    assert sdk.headers == original


def test_new_default_request_id_does_not_reuse_client_header(sdk, http):
    sdk.headers["x-request-id"] = REQUEST
    serve(http[0], {"id": TARGET, "name": "Fork"})
    sdk.fork_conversation(SOURCE, MESSAGE)
    assert http[0].call_args.kwargs["headers"]["X-Request-ID"] != REQUEST
    assert sdk.headers["x-request-id"] == REQUEST


def test_inherited_scope_is_validated(sdk, http):
    sdk.headers.update({"x-agent-id": AGENT, "x-company-id": COMPANY})
    data = receipt("completed")
    data["transfer"]["company_id"] = TARGET
    serve(http[0], data)
    with pytest.raises(ProjectTransferError):
        sdk.fork_conversation(SOURCE, MESSAGE, request_id=REQUEST)
    http[0].assert_called_once()
    http[1].assert_not_called()


def test_source_uuid_spelling_does_not_change_identity(sdk, http):
    serve(http[0], receipt("completed"))
    assert (
        sdk.fork_conversation(SOURCE.upper(), MESSAGE, request_id=REQUEST)["id"]
        == TARGET
    )


def test_concurrent_calls_have_isolated_stable_headers(sdk, http):
    barrier = Barrier(2)
    original = deepcopy(sdk.headers)

    def submit(**kwargs):
        headers = kwargs["headers"]
        before = dict(headers)
        barrier.wait(timeout=5)
        assert dict(headers) == before
        return response({"id": TARGET, "name": "Legacy fork"})

    http[0].side_effect = submit
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(sdk.fork_conversation, SOURCE, MESSAGE, agent_id=agent)
            for agent in (AGENT, TARGET)
        ]
        for future in futures:
            assert future.result()["id"] == TARGET
    headers = [call.kwargs["headers"] for call in http[0].call_args_list]
    assert headers[0] is not headers[1]
    assert headers[0]["X-Request-ID"] != headers[1]["X-Request-ID"]
    assert {item["X-Agent-ID"] for item in headers} == {AGENT, TARGET}
    assert sdk.headers == original
    http[1].assert_not_called()
