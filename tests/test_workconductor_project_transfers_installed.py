"""Parent-serialized, opt-in proof of the maintained SDK against an installed API.

Run only after deployment, as root, with WC_SDK_INSTALLED_TEST=1,
WC_WORKCONDUCTOR_CHECKOUT=<absolute checkout>, WC_COMPUTER_API_IMAGE=sha256:...,
WC_COMPUTER_RUNTIME_BUNDLE=<absolute bundle>, and WC_COMPUTER_MANIFEST_SHA256.
Default discovery imports neither WorkConductor nor the SDK and touches no API.
No build, live scope, installed SDK modification, or automatic failure cleanup.
The inherited fixture owns all fake users, Postgres, API, home, and host resources.
Bounds: 600 seconds overall, two forks, 90-second observations, 5-second HTTP,
one 256-byte binary, and a private proof of at most 32 KiB. Not a quota matrix.
"""

from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
import unittest
import uuid


SDK_ROOT = Path(__file__).resolve().parents[1]
OPT_IN = "WC_SDK_INSTALLED_TEST"
BINARY = bytes(range(256))
FILENAME = "sdk-owned-binary.dat"
CONTENT = "Synthetic owned SDK fork boundary"


def valid_id(value):
    parsed = uuid.UUID(value)
    assert parsed.int and str(parsed) == value, "Invalid fixture identity"
    return value


def preflight():
    assert os.environ.get(OPT_IN) == "1", "Explicit installed SDK opt-in required"
    assert os.geteuid() == 0, "Parent-reviewed root/systemd invocation required"
    checkout = Path(os.environ.get("WC_WORKCONDUCTOR_CHECKOUT", ""))
    assert checkout.is_absolute() and checkout.is_dir(), "Checkout required"
    checkout = checkout.resolve(strict=True)
    assert (checkout / "scripts/tests/test_installed_user_computer_api.py").is_file()
    assert re.fullmatch(
        r"sha256:[0-9a-f]{64}", os.environ.get("WC_COMPUTER_API_IMAGE", "")
    ), "Immutable API image required"
    assert re.fullmatch(
        r"[0-9a-f]{64}", os.environ.get("WC_COMPUTER_MANIFEST_SHA256", "")
    ), "Manifest pin required"
    bundle = Path(os.environ.get("WC_COMPUTER_RUNTIME_BUNDLE", ""))
    assert bundle.is_absolute() and bundle.is_dir(), "Runtime bundle required"
    assert not any(
        value
        for key, value in os.environ.items()
        if key.startswith(
            (
                "WC_COMPUTER_HOST_DEBUG_",
                "WC_COMPUTER_PROVISION_DEBUG_",
                "WC_RECLAMATION_",
            )
        )
        or key
        in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")
    ), "Debug, rollback, or alternate Docker overrides refused"
    return checkout


@contextmanager
def source_imports(checkout):
    original = sys.path[:]
    # Requests must reach the owned bridge API directly, never an ambient proxy.
    proxy = {key: os.environ.get(key) for key in ("NO_PROXY", "no_proxy")}
    sys.path[:0] = [str(SDK_ROOT), str(checkout)]
    for key in proxy:
        os.environ[key] = "*"
    try:
        yield
    finally:
        sys.path[:] = original
        for key, value in proxy.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def effects(f):
    return f.sql_json(
        "SELECT json_build_object("
        "'transfers',(SELECT count(*) FROM user_computer_project_transfers),"
        "'imports',(SELECT count(*) FROM user_computer_project_imports),"
        "'conversations',(SELECT count(*) FROM conversation),"
        "'projects',(SELECT count(*) FROM user_computer_projects),"
        "'bindings',(SELECT count(*) FROM user_computer_project_conversations),"
        "'messages',(SELECT count(*) FROM message))"
    )


def history(f, conversation):
    return f.sql_json(
        "SELECT COALESCE(json_agg(json_build_object('role',role,'content',content) "
        "ORDER BY timestamp ASC,transfer_order ASC NULLS LAST,ctid ASC),'[]'::json) "
        f"FROM message WHERE conversation_id='{valid_id(conversation)}'"
    )


def complete_identity(value, request, source, agent, company):
    assert isinstance(value, dict), "Missing SDK receipt"
    transfer = value["transfer"]
    assert transfer["state"] == "completed", "Fork did not complete within its bound"
    assert transfer["request_id"] == request
    assert transfer["source_conversation_id"] == source
    assert transfer["agent_id"] == agent and transfer["company_id"] == company
    assert transfer["error"] is None
    target = valid_id(value["id"])
    assert target == transfer["conversation_id"] and target != source
    assert isinstance(value["name"], str) and 0 < len(value["name"]) <= 1024
    assert transfer["summary"]["files"] == 1
    assert transfer["summary"]["bytes"] == len(BINARY)
    assert transfer["summary"]["repositories"] == 0
    return target


def observe_pending(sdk, value, request, source, agent, company):
    until = time.monotonic() + 90
    while value["transfer"]["state"] in ("queued", "running"):
        remaining = until - time.monotonic()
        assert remaining > 0, "Legacy fork observation deadline exceeded"
        value = sdk.get_project_transfer(
            request,
            agent,
            company,
            source_conversation_id=source,
            timeout=min(5, remaining),
        )
        if value["transfer"]["state"] in ("queued", "running"):
            time.sleep(min(0.25, max(0, until - time.monotonic())))
    return value


def verify_copy(f, target, source, message, expected_history):
    imported = f.sql_json(
        "SELECT json_build_object('project_id',i.project_id,'binding',b.project_id) "
        "FROM user_computer_project_imports i "
        "JOIN user_computer_project_conversations b ON b.conversation_id=i.conversation_id "
        f"WHERE i.computer_id='{valid_id(f.computer['computer_id'])}' "
        f"AND i.conversation_id='{valid_id(target)}'"
    )
    project = valid_id(imported["project_id"])
    assert imported["binding"] == project and project not in (*f.projects, target)
    assert history(f, target) == expected_history
    assert history(f, source) == expected_history
    assert f.sql_json(
        "SELECT json_build_object('role',role,'content',content) FROM message "
        f"WHERE id='{valid_id(message)}' AND conversation_id='{source}'"
    ) == {"role": "USER", "content": CONTENT}
    assert (
        f.sql_json(
            "SELECT to_json(project_id) FROM user_computer_project_conversations "
            f"WHERE conversation_id='{source}'"
        )
        == source
    ), "Original native project binding changed"
    original = Path(f.computer["home"]) / "projects" / source / FILENAME
    assert original.read_bytes() == BINARY, "Source bytes changed"
    for conversation in (source, target):
        code, data = f.http(
            "GET",
            f"/v1/conversation/{conversation}/workspace/download?path={FILENAME}",
            user=f.users[0],
            raw_response=True,
        )
        assert code == 200 and isinstance(data, bytes), "Binary download failed"
        assert len(data) == len(BINARY)
        assert hashlib.sha256(data).digest() == hashlib.sha256(BINARY).digest()
    return project


def exercise(f, sdk_module):
    source = valid_id(f.projects[0])
    company = valid_id(f.company)
    user = valid_id(f.users[0]["user_id"])
    agents = f.sql_json(
        "SELECT COALESCE(json_agg(json_build_object('id',id,'name',name)),'[]'::json) "
        f"FROM agent WHERE user_id::text='{user}'"
    )
    default = [row for row in agents if row["name"] == "XT"]
    assert len(default) == 1, "Expected registered user's actual default agent"
    agent = valid_id(default[0]["id"])
    message = str(uuid.uuid4())
    f.sql(
        "INSERT INTO conversation_participant(id,conversation_id,agent_id,participant_type,role,status,notification_mode) "
        f"SELECT '{uuid.uuid4()}','{source}','{agent}','agent','member','active','all' "
        f"WHERE NOT EXISTS (SELECT 1 FROM conversation_participant WHERE conversation_id='{source}' AND agent_id='{agent}');"
        "INSERT INTO message(id,conversation_id,role,content,timestamp,feedback_received,notify,pinned) "
        f"VALUES('{message}','{source}','USER','{CONTENT}',clock_timestamp(),false,false,false);"
    )
    expected_history = [{"role": "USER", "content": CONTENT}]
    assert history(f, source) == expected_history
    home = Path(f.computer["home"]).resolve(strict=True)
    directory = (home / "projects" / source).resolve(strict=True)
    assert home.is_relative_to(f.root) and directory.is_relative_to(home)
    binary = directory / FILENAME
    with binary.open("xb") as file:
        file.write(BINARY)
    os.chmod(binary, 0o600)
    os.chown(binary, 1000, 1000)
    sdk = sdk_module.AGiXTSDK(base_uri=f.base, api_key=f.users[0]["token"])
    baseline = effects(f)
    assert baseline["transfers"] == baseline["imports"] == 0
    assert baseline["projects"] == baseline["bindings"] == 2

    preview = sdk.preview_conversation_fork(
        source, agent_id=agent, company_id=company, timeout=5
    )
    assert preview["source_conversation_id"] == preview["project_id"] == source
    assert (
        preview["repositories"] == []
    ), "Legacy proof requires no associated repository"
    assert preview["project_files_included"] is True
    assert preview["credentials_included"] is False
    assert preview["repository_metadata_included"] is False
    assert re.fullmatch(r"[0-9a-f]{64}", preview["preview_digest"])
    assert effects(f) == baseline, "Preview created effects"
    request, legacy_request = str(uuid.uuid4()), str(uuid.uuid4())
    options = {
        "request_id": request,
        "agent_id": agent,
        "company_id": company,
        "preview_digest": preview["preview_digest"],
        "repositories": [],
        "timeout": 90,
        "poll_interval": 0.25,
        "request_timeout": 5,
    }
    receipt = sdk.fork_conversation(source, message, **options)
    target = complete_identity(receipt, request, source, agent, company)
    after_selected = effects(f)
    assert after_selected == {key: value + 1 for key, value in baseline.items()}
    project = verify_copy(f, target, source, message, expected_history)

    f.restart_api()
    sdk.base_uri = f.base
    observed = sdk.get_project_transfer(
        request, agent, company, source_conversation_id=source, timeout=5
    )
    assert observed == receipt, "Original receipt changed across API restart"
    replay = sdk.fork_conversation(source, message, **options)
    assert (
        replay == receipt and replay["id"] == target
    ), "Replay created a different fork"
    assert effects(f) == after_selected, "Replay created extra database effects"
    assert verify_copy(f, target, source, message, expected_history) == project

    pending = sdk.fork_conversation(
        source,
        message,
        request_id=legacy_request,
        agent_id=agent,
        company_id=company,
        wait=False,
        request_timeout=5,
    )
    legacy_was_pending = pending["transfer"]["state"] in ("queued", "running")
    legacy = observe_pending(sdk, pending, legacy_request, source, agent, company)
    legacy_target = complete_identity(legacy, legacy_request, source, agent, company)
    assert legacy_target not in (*f.projects, target)
    legacy_project = verify_copy(f, legacy_target, source, message, expected_history)
    assert legacy_project != project
    final = effects(f)
    assert final == {key: value + 2 for key, value in baseline.items()}

    for foreign in (f.users[2], f.users[1]):
        other = sdk_module.AGiXTSDK(base_uri=f.base, api_key=foreign["token"])
        for request_id in (request, legacy_request):
            denied = other.get_project_transfer(
                request_id, agent, company, source_conversation_id=source, timeout=5
            )
            assert isinstance(denied, dict) and set(denied) == {"detail"}
            encoded = json.dumps(denied)
            assert len(encoded) <= 4096
            assert not any(
                private in encoded
                for private in (source, target, legacy_target, CONTENT, FILENAME)
            ), "Foreign SDK status exposed owned transfer/content"
    assert effects(f) == final, "Read-only observation created effects"
    return {
        "source_conversation_id": source,
        "original_project_id": source,
        "agent_id": agent,
        "company_id": company,
        "message_id": message,
        "completed": [
            {"request_id": request, "id": target, "project_id": project},
            {
                "request_id": legacy_request,
                "id": legacy_target,
                "project_id": legacy_project,
            },
        ],
        "effects": {"before": baseline, "selected": after_selected, "final": final},
        "reviewed_repositories": [],
        "binary_sha256": hashlib.sha256(BINARY).hexdigest(),
        "binary_bytes": len(BINARY),
        "restart_receipt_exact": True,
        "replay_receipt_exact_no_extra_effects": True,
        "source_message_and_bytes_intact": True,
        "legacy_wait_false_was_pending": legacy_was_pending,
        "foreign_same_company_and_cross_company_observations_denied": True,
    }


def owned_containers(f):
    owned = set(getattr(f, "resources", []))
    if getattr(f, "computer", None):
        owned.add(f.computer["container_id"])
    return owned


@unittest.skipUnless(os.environ.get(OPT_IN) == "1", "opt-in installed SDK acceptance")
class InstalledProjectTransferTests(unittest.TestCase):
    def test_maintained_sdk_native_forks(self):
        f = None
        api = None
        step = "preflight"
        started = time.monotonic()
        try:
            checkout = preflight()
            with source_imports(checkout):
                # All checkout imports are deliberately below the opt-in/root gates.
                api = importlib.import_module(
                    "scripts.tests.test_installed_user_computer_api"
                )
                preservation = importlib.import_module(
                    "scripts.tests.test_installed_user_computer_repository_preparing_reclamation"
                )
                documents = importlib.import_module(
                    "scripts.tests.test_installed_user_computer_snapshot_documents"
                )
                for module in (api, preservation, documents):
                    assert Path(module.__file__).resolve() == (
                        checkout / Path(*module.__name__.split("."))
                    ).with_suffix(".py"), "WorkConductor imported from another checkout"
                with documents.finite_run(seconds=600):
                    sdk_module = importlib.import_module("agixtsdk")
                    sdk_source = SDK_ROOT / "agixtsdk/__init__.py"
                    assert Path(sdk_module.__file__).resolve() == sdk_source
                    requests = importlib.import_module("requests")
                    assert sdk_module.requests is requests
                    assert requests.get is requests.api.get
                    assert requests.post is requests.api.post
                    source_hash = hashlib.sha256(sdk_source.read_bytes()).hexdigest()
                    before = documents.containers()
                    step = "owned fixture setup"
                    # Retain the instance even when its constructor fails midway.
                    f = api.InstalledApiFixture.__new__(api.InstalledApiFixture)
                    f.__init__()
                    step = "SDK fork/restart/replay/isolation checks"
                    evidence = exercise(f, sdk_module)
                    step = "API diagnostics and container preservation"
                    f.capture_api_diagnostics(f.api)
                    preservation.assert_preserved(
                        before, documents.containers(), owned_containers(f)
                    )
                    assert (
                        hashlib.sha256(sdk_source.read_bytes()).hexdigest()
                        == source_hash
                    )
                    proof = {
                        "kind": "maintained-sdk-installed-project-transfer-adapter",
                        "workconductor_checkout": str(checkout),
                        "fixture_source_sha256": hashlib.sha256(
                            Path(api.__file__).read_bytes()
                        ).hexdigest(),
                        "api_image": f.api_image,
                        "runtime_bundle": str(f.bundle),
                        "manifest_sha256": f.manifest_sha256,
                        "runtime_image": f.install["image_id"],
                        "sdk_source": str(sdk_source),
                        "sdk_source_sha256": source_hash,
                        **evidence,
                        # Raw log lines can contain credentials; record counts only.
                        "api_diagnostics": [
                            {
                                "errors": len(row["errors"]),
                                "warnings": len(row["warnings"]),
                                "retryable_responses": row["retryable_responses"],
                            }
                            for row in f.api_diagnostics
                        ],
                        "protected_containers": len(before),
                    }
                    path = Path("/tmp") / (
                        "wc-sdk-installed-proof-" + f.identity + ".json"
                    )
                    assert not path.exists(), "Exclusive proof path required"
                    step = "ownership-validated success cleanup"
                    f.cleanup()
                    preservation.assert_preserved(before, documents.containers())
                    proof.update(
                        owned_resources_cleaned=True,
                        preexisting_containers_preserved=True,
                        elapsed_seconds=round(time.monotonic() - started, 3),
                    )
                    assert len(json.dumps(proof, indent=2).encode()) <= 32 * 1024
                    api.private_json(path, proof)
                    assert path.stat().st_mode & 0o777 == 0o600
                    assert path.stat().st_size <= 32 * 1024
                    print("Installed SDK acceptance proof:", path, flush=True)
        except BaseException:
            # Never expose inherited exceptions, HTTP bodies, JWTs, or DB URLs.
            root = getattr(f, "root", None)
            retained = str(root) if root is not None else "not created"
            if root is not None and root.is_dir() and api is not None:
                failure = {
                    "stage": step,
                    "identity": f.identity,
                    "root": str(root),
                    "containers": sorted(owned_containers(f)),
                    "unit": getattr(f, "unit", None),
                    "api": getattr(f, "api", None),
                    "postgres": getattr(f, "pg", None),
                    "cleanup_attempted": step == "ownership-validated success cleanup",
                }
                try:
                    assert len(json.dumps(failure, indent=2).encode()) <= 32 * 1024
                    api.private_json(root / "sdk-failure-resources.json", failure)
                except (OSError, AssertionError):
                    pass
            raise AssertionError(
                f"Installed SDK acceptance failed during {step}; "
                f"no failure cleanup attempted; fixture root: {retained}"
            ) from None


if __name__ == "__main__":
    unittest.main()
