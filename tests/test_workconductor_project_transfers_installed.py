"""Parent-serialized, opt-in proof of the maintained SDK against an installed API.

Run only after deployment, as root, with WC_SDK_INSTALLED_TEST=1,
WC_WORKCONDUCTOR_CHECKOUT=<absolute checkout>, WC_COMPUTER_API_IMAGE=sha256:...,
WC_COMPUTER_RUNTIME_BUNDLE=<absolute bundle>, and WC_COMPUTER_MANIFEST_SHA256.
Default discovery imports neither WorkConductor nor the SDK and touches no API.
No build, live scope, installed SDK modification, or automatic failure cleanup.
The inherited fixture owns all fake users, Postgres, API, home, and host resources.
Bounds per case: 600 seconds overall, at most two forks, 90-second observations,
5-second HTTP/Git, synthetic repositories, and a private proof of at most 32 KiB.
The separate selected case copies one populated checkout and omits one sibling.
Not a quota matrix. No Git remotes, credentials, clones, or network are needed.
"""

from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unittest
import uuid
from urllib.parse import urlencode


SDK_ROOT = Path(__file__).resolve().parents[1]
OPT_IN = "WC_SDK_INSTALLED_TEST"
BINARY = bytes(range(256))
FILENAME = "sdk-owned-binary.dat"
CONTENT = "Synthetic owned SDK fork boundary"
REPOSITORY_FILES = {
    "tracked.txt": b"owned dirty repository bytes\n",
    "untracked.bin": BINARY,
}
SIBLING_PATH = "sdk-unselected-sibling"
SIBLING_FILE = "sdk-unselected-only.txt"
SIBLING_BYTES = b"owned deliberately unselected sibling\n"


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


def complete_identity(
    value, request, source, agent, company, *, repositories=0, files=1, size=len(BINARY)
):
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
    assert transfer["summary"]["files"] == files
    assert transfer["summary"]["bytes"] == size
    assert transfer["summary"]["repositories"] == repositories
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


def prepare_source(f):
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
    return source, company, agent, message, expected_history


def exercise(f, sdk_module):
    source, company, agent, message, expected_history = prepare_source(f)
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


def repository_inventory(f, conversation, project, generation):
    uri = f"/v1/conversation/{valid_id(conversation)}/workspace/repositories?"
    uri += urlencode({"project_id": valid_id(project), "generation": generation})
    code, value = f.http("GET", uri, user=f.users[0])
    assert code == 200 and value["metadata_only"] is True
    assert value["project_id"] == project and value["generation"] == generation
    assert value["next_cursor"] is None, "Owned inventory must fit one page"
    return value["checkouts"]


def repository_effects(f):
    return {
        **effects(f),
        **f.sql_json(
            "SELECT json_build_object("
            "'repositories',(SELECT count(*) FROM user_computer_repositories),"
            "'checkouts',(SELECT count(*) FROM user_computer_repository_checkouts),"
            "'associations',(SELECT count(*) FROM user_computer_project_repositories))"
        ),
    }


def seed_repository(home, directory, files):
    assert not os.path.lexists(directory), "Exclusive owned checkout required"
    parent = home
    for part in directory.relative_to(home).parts:
        parent = parent / part
        assert not parent.is_symlink(), "Owned repository path must not follow links"
        if not parent.exists():
            parent.mkdir(mode=0o700)
            # Keep the checkout owned by the seeding process until Git finishes.
            if parent != directory:
                os.chown(parent, 1000, 1000)
        assert parent.is_dir()
    # Ignore all ambient Git configuration/templates/hooks; no remote is created.
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(directory),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
    }
    for arguments in (
        ["init", "-q", "--initial-branch=main", "--template="],
        ["add", "tracked.txt"],
        [
            "-c",
            "user.name=Owned SDK fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "owned offline seed",
        ],
    ):
        if arguments[0] == "add":
            (directory / "tracked.txt").write_bytes(b"owned committed seed\n")
        result = subprocess.run(
            ["/usr/bin/git", "-C", str(directory), "-c", "core.hooksPath=/dev/null"]
            + arguments,
            env=env,
            capture_output=True,
            timeout=5,
            check=False,
        )
        assert result.returncode == 0, "Owned offline Git seed failed"
    for name, data in files.items():
        with (directory / name).open("xb" if name != "tracked.txt" else "wb") as out:
            out.write(data)
    (directory / ".git/sdk-source-only").write_bytes(b"owned Git metadata\n")
    paths = [directory, *directory.rglob("*")]
    assert len(paths) <= 128 and not any(path.is_symlink() for path in paths)
    assert sum(path.stat().st_size for path in paths if path.is_file()) <= 1024 * 1024
    for path in paths:
        path.chmod(0o700 if path.is_dir() else 0o600)
        os.chown(path, 1000, 1000)
    return (directory / ".git/refs/heads/main").read_bytes()


def exercise_selected(f, sdk_module):
    source, company, agent, message, expected_history = prepare_source(f)
    code, storage = f.http(
        "GET", f"/v1/conversation/{source}/workspace/storage", user=f.users[0]
    )
    assert code == 200 and storage["mode"] == "computer"
    assert storage["conversation_id"] == storage["project_id"] == source
    generation = storage["generation"]
    assert isinstance(generation, int) and generation > 0
    home = Path(f.computer["home"]).resolve(strict=True)
    source_rows, roots, commits = [], [], []
    # Register exact checkout locations using the ordinary public repository API.
    # The shared checkout is selected; the embedded sibling must not leak through
    # the otherwise-included project tree.
    for selected in (True, False):
        repository, checkout = str(uuid.uuid4()), str(uuid.uuid4())
        location = (
            {"kind": "shared"}
            if selected
            else {"kind": "project", "project_id": source, "path": SIBLING_PATH}
        )
        root = (
            home / "repos" / repository / "checkouts" / checkout
            if selected
            else home / "projects" / source / SIBLING_PATH
        )
        assert root.is_relative_to(home) and home.is_relative_to(f.root)
        files = REPOSITORY_FILES if selected else {SIBLING_FILE: SIBLING_BYTES}
        commits.append(seed_repository(home, root, files))
        roots.append(root)
        declaration = {
            "request_id": str(uuid.uuid4()),
            "project_id": source,
            "conversation_id": source,
            "generation": generation,
            "change": {
                "action": "register",
                "repository_id": repository,
                "checkout_id": checkout,
                "display_name": "Owned selected" if selected else "Owned omitted",
                "location": location,
                "source_checkout_id": None,
            },
        }
        code, registered = f.http(
            "POST",
            f"/v1/conversation/{source}/workspace/repositories",
            declaration,
            user=f.users[0],
        )
        assert code == 200 and registered["recorded"] is True
        assert registered["metadata_only"] is True
        assert registered["request"] == declaration
        source_rows.append(declaration["change"])
    original_rows = repository_inventory(f, source, source, generation)
    assert len(original_rows) == 2
    assert {row["checkout_id"] for row in original_rows} == {
        row["checkout_id"] for row in source_rows
    }
    for expected in source_rows:
        actual = next(
            row
            for row in original_rows
            if row["checkout_id"] == expected["checkout_id"]
        )
        assert all(
            actual[key] == expected[key] for key in ("repository_id", "location")
        )
        assert actual["associated_project_ids"] == [source]
    sdk = sdk_module.AGiXTSDK(base_uri=f.base, api_key=f.users[0]["token"])
    baseline = repository_effects(f)
    assert baseline["transfers"] == baseline["imports"] == 0
    preview = sdk.preview_conversation_fork(
        source, agent_id=agent, company_id=company, timeout=5
    )
    assert preview["source_conversation_id"] == preview["project_id"] == source
    assert preview["generation"] == generation
    choices = {row["checkout_id"]: row for row in preview["repositories"]}
    assert set(choices) == {row["checkout_id"] for row in source_rows}
    chosen, omitted = [row["checkout_id"] for row in source_rows]
    assert choices[chosen]["included_in_project"] is False
    assert choices[omitted]["included_in_project"] is True
    assert not any(row["required"] for row in choices.values())
    assert preview["project_files_included"] is True
    assert preview["credentials_included"] is False
    assert preview["repository_metadata_included"] is False
    assert re.fullmatch(r"[0-9a-f]{64}", preview["preview_digest"])
    assert repository_effects(f) == baseline, "Preview created effects"
    request = str(uuid.uuid4())
    options = {
        "request_id": request,
        "agent_id": agent,
        "company_id": company,
        "preview_digest": preview["preview_digest"],
        "repositories": [chosen],
        "timeout": 90,
        "poll_interval": 0.25,
        "request_timeout": 5,
    }
    # These POSTs must use the maintained SDK with real requests, never f.http.
    receipt = sdk.fork_conversation(source, message, **options)
    target = complete_identity(
        receipt,
        request,
        source,
        agent,
        company,
        repositories=1,
        files=1 + len(REPOSITORY_FILES),
        size=len(BINARY) + sum(map(len, REPOSITORY_FILES.values())),
    )
    project = verify_copy(f, target, source, message, expected_history)
    after = repository_effects(f)
    assert after == {key: value + 1 for key, value in baseline.items()}
    targets = repository_inventory(f, target, project, generation)
    assert len(targets) == 1, "Unselected sibling was associated with target"
    row = targets[0]
    fresh = [valid_id(row[key]) for key in ("repository_id", "checkout_id")]
    source_ids = {
        value[key] for value in source_rows for key in ("repository_id", "checkout_id")
    }
    assert len(set(fresh + [project, target])) == 4
    assert (source_ids | set(f.projects)).isdisjoint(fresh + [project, target])
    assert row["location"] == {"kind": "shared"}
    assert row["associated_project_ids"] == [project]
    repository_root = home / "repos" / fresh[0] / "checkouts" / fresh[1]
    assert row["guest_path"] == f"/home/agent/repos/{fresh[0]}/checkouts/{fresh[1]}"
    assert repository_root.is_dir() and (repository_root / ".git").is_dir()
    assert not (repository_root / ".git/sdk-source-only").exists()

    def verify_repositories():
        assert repository_inventory(f, target, project, generation) == targets
        assert repository_inventory(f, source, source, generation) == original_rows
        for name, data in REPOSITORY_FILES.items():
            assert (roots[0] / name).read_bytes() == data
            assert (repository_root / name).read_bytes() == data
            uri = f"/v1/conversation/{target}/workspace/repositories/{fresh[1]}/file?"
            uri += urlencode(
                {"project_id": project, "generation": generation, "path": name}
            )
            code, copied = f.http("GET", uri, user=f.users[0], raw_response=True)
            assert code == 200 and copied == data, "Copied repository download differs"
            for foreign in (f.users[2], f.users[1]):
                assert f.http("GET", uri, user=foreign, raw_response=True)[0] in (
                    403,
                    404,
                )
        assert (roots[1] / SIBLING_FILE).read_bytes() == SIBLING_BYTES
        for root, commit in zip(roots, commits):
            assert (root / ".git/refs/heads/main").read_bytes() == commit
            assert (
                root / ".git/sdk-source-only"
            ).read_bytes() == b"owned Git metadata\n"
        target_project = home / "projects" / project
        assert not os.path.lexists(target_project / SIBLING_PATH)
        assert not list(target_project.rglob(SIBLING_FILE))
        assert not list(repository_root.rglob(SIBLING_FILE))
        code, _ = f.http(
            "GET",
            f"/v1/conversation/{target}/workspace/download?"
            + urlencode({"path": f"{SIBLING_PATH}/{SIBLING_FILE}"}),
            user=f.users[0],
            raw_response=True,
        )
        assert code == 404, "Unselected repository leaked via project download"

    verify_repositories()
    assert sdk.fork_conversation(source, message, **options) == receipt
    assert repository_effects(f) == after, "Replay duplicated target or repository"
    assert verify_copy(f, target, source, message, expected_history) == project
    verify_repositories()
    private_values = (
        source,
        target,
        project,
        *source_ids,
        *fresh,
        CONTENT,
        FILENAME,
        SIBLING_FILE,
        REPOSITORY_FILES["tracked.txt"].decode().strip(),
    )
    for foreign in (f.users[2], f.users[1]):
        other = sdk_module.AGiXTSDK(base_uri=f.base, api_key=foreign["token"])
        for denied in (
            other.preview_conversation_fork(
                source, agent_id=agent, company_id=company, timeout=5
            ),
            other.get_project_transfer(
                request, agent, company, source_conversation_id=source, timeout=5
            ),
            other.fork_conversation(source, message, **options),
        ):
            assert isinstance(denied, dict) and set(denied) == {"detail"}
            encoded = json.dumps(denied)
            assert len(encoded) <= 4096
            assert not any(value in encoded for value in private_values)
    assert repository_effects(f) == after, "Foreign denial created effects"
    return {
        "source_conversation_id": source,
        "agent_id": agent,
        "company_id": company,
        "message_id": message,
        "completed": [{"request_id": request, "id": target, "project_id": project}],
        "reviewed_repositories": [chosen, omitted],
        "selected_repositories": [chosen],
        "target_repository_id": fresh[0],
        "target_checkout_id": fresh[1],
        "repository_files_sha256": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in REPOSITORY_FILES.items()
        },
        "repository_bytes": sum(map(len, REPOSITORY_FILES.values())),
        "binary_sha256": hashlib.sha256(BINARY).hexdigest(),
        "binary_bytes": len(BINARY),
        "effects": {"before": baseline, "final": after},
        "replay_receipt_exact_no_extra_effects": True,
        "source_message_and_bytes_intact": True,
        "unselected_sibling_absent_from_target_registry_and_bytes": True,
        "foreign_same_company_and_cross_company_preview_post_status_and_files_denied": True,
    }


def diagnostic_counts(row):
    categories = {"schema_or_seeding": 0, "redis": 0, "unknown": 0}
    for line in row["warnings"]:
        if any(
            marker in line
            for marker in ("Failed to create", "schema parity", "seeding failed")
        ):
            category = "schema_or_seeding"
        elif "Redis" in line or "redis" in line:
            category = "redis"
        else:
            category = "unknown"
        categories[category] += 1
    return {
        "errors": len(row["errors"]),
        "warnings": len(row["warnings"]),
        "warning_categories": categories,
        "warning_review_required": bool(row["warnings"]),
        "retryable_responses": row["retryable_responses"],
    }


def owned_containers(f):
    owned = set(getattr(f, "resources", []))
    if getattr(f, "computer", None):
        owned.add(f.computer["container_id"])
    return owned


@unittest.skipUnless(os.environ.get(OPT_IN) == "1", "opt-in installed SDK acceptance")
class InstalledProjectTransferTests(unittest.TestCase):
    def test_maintained_sdk_native_forks(self):
        self.run_installed_case(
            exercise, "maintained-sdk-installed-project-transfer-adapter"
        )

    def test_maintained_sdk_selected_repository_fork(self):
        self.run_installed_case(
            exercise_selected, "maintained-sdk-installed-selected-repository-adapter"
        )

    def run_installed_case(self, exercise_case, kind):
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
                    step = "SDK fork/replay/isolation checks"
                    evidence = exercise_case(f, sdk_module)
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
                        "kind": kind,
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
                            diagnostic_counts(row) for row in f.api_diagnostics
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
