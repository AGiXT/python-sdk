# AGiXT SDK for Python

[![GitHub](https://img.shields.io/badge/GitHub-Sponsor%20Josh%20XT-blue?logo=github&style=plastic)](https://github.com/sponsors/Josh-XT) [![PayPal](https://img.shields.io/badge/PayPal-Sponsor%20Josh%20XT-blue.svg?logo=paypal&style=plastic)](https://paypal.me/joshxt) [![Ko-Fi](https://img.shields.io/badge/Kofi-Sponsor%20Josh%20XT-blue.svg?logo=kofi&style=plastic)](https://ko-fi.com/joshxt)

[![GitHub](https://img.shields.io/badge/GitHub-AGiXT%20Core-blue?logo=github&style=plastic)](https://github.com/Josh-XT/AGiXT) [![GitHub](https://img.shields.io/badge/GitHub-AGiXT%20Hub-blue?logo=github&style=plastic)](https://github.com/AGiXT/hub) [![GitHub](https://img.shields.io/badge/GitHub-AGiXT%20NextJS%20Web%20UI-blue?logo=github&style=plastic)](https://github.com/AGiXT/nextjs) [![GitHub](https://img.shields.io/badge/GitHub-AGiXT%20Streamlit%20Web%20UI-blue?logo=github&style=plastic)](https://github.com/AGiXT/streamlit)

[![GitHub](https://img.shields.io/badge/GitHub-AGiXT%20Python%20SDK-blue?logo=github&style=plastic)](https://github.com/AGiXT/python-sdk) [![pypi](https://img.shields.io/badge/pypi-AGiXT%20Python%20SDK-blue?logo=pypi&style=plastic)](https://pypi.org/project/agixtsdk/)

[![GitHub](https://img.shields.io/badge/GitHub-AGiXT%20TypeScript%20SDK-blue?logo=github&style=plastic)](https://github.com/AGiXT/typescript-sdk) [![npm](https://img.shields.io/badge/npm-AGiXT%20TypeScript%20SDK-blue?logo=npm&style=plastic)](https://www.npmjs.com/package/agixt)

[![GitHub](https://img.shields.io/badge/GitHub-AGiXT%20Dart%20SDK-blue?logo=github&style=plastic)](https://github.com/AGiXT/dart-sdk)

[![Discord](https://img.shields.io/discord/1097720481970397356?label=Discord&logo=discord&logoColor=white&style=plastic&color=5865f2)](https://discord.gg/d3TkHRZcjD)
[![Twitter](https://img.shields.io/badge/Twitter-Follow_@Josh_XT-blue?logo=twitter&style=plastic)](https://twitter.com/Josh_XT)

![AGiXT_New](https://github.com/user-attachments/assets/14a5c1ae-6af8-4de8-a82e-f24ea52da23f)


This repository is for the [AGiXT](https://github.com/Josh-XT/AGiXT) SDK for Python.

## Installation

```bash
pip install agixtsdk
```

## Usage

```python
from agixtsdk import AGiXTSDK

base_uri = "http://localhost:7437"
api_key = "your_agixt_api_key"

ApiClient = AGiXTSDK(base_uri=base_uri, api_key=api_key)
```

Check out the AGiXT [Examples and Tests Notebook](https://github.com/Josh-XT/AGiXT/blob/main/tests/endpoint-tests.ipynb) for examples of how to use the AGiXT SDK for Python.

### Conversation Forks

`fork_conversation(conversation_id, message_id)` keeps its positional arguments
and legacy JSON return. Each call sends a fresh `X-Request-ID` UUID unless a
keyword `request_id` is supplied; headers are copied, never changed on the client.
No selection uses the legacy fork URL, including native project forks with no
associated repositories. Optional `agent_id` and `company_id` set per-call scope.
The legacy `"-"` source alias is sent unchanged in that single POST, without a
lookup. A native receipt resolves it to a canonical source UUID, which is pinned
for all status polling. Explicit repository selection requires a source UUID,
not `"-"`.

For associated repositories, explicitly review and select checkout IDs:

```python
import uuid

preview = ApiClient.preview_conversation_fork(conversation_id, company_id=company_id)
request_id = str(uuid.uuid4())  # Retain for read-only recovery after connection loss.
result = ApiClient.fork_conversation(
    conversation_id,
    message_id,
    request_id=request_id,
    agent_id=agent_id,
    company_id=company_id,
    preview_digest=preview["preview_digest"],
    repositories=selected_checkout_ids,
)
```

Both selection keywords are required together and use the project-transfer URL.
`repositories=[]` explicitly omits associated repositories; the server can refuse
omitting a required project-root repository. The SDK never selects all, refreshes
a digest, or falls back to an unselected POST.

Queued/running transfers are polled with GET only, using the original credentials
and exact request/agent/company. Completed responses include legacy `id`/`name`
and the transfer receipt. `attention`/`failed` outcomes are returned unchanged,
without resume or regrant. Set `wait=False` for the immediate receipt; `timeout=60`
seconds bounds the default observation window, `poll_interval=2` controls polling,
and `request_timeout=30` bounds each HTTP call. Exhausting the observation window
returns the original pending receipt, not a fabricated completion.

Network/JSON failures raise `agixtsdk.ProjectTransferError` with a safe
`request_id` attribute, without including credential-bearing exception details.
Redirects are disabled for every fork/transfer HTTP call; any 3xx raises a safe
`ProjectTransferError` without exposing the redirect target or response body.
No POST is retried automatically. Recover by calling
`ApiClient.get_project_transfer(request_id, agent_id, company_id,
source_conversation_id=conversation_id)` with the original scope. This read-only
helper validates returned request/agent/company and, when supplied, source identity;
fork polling always validates the source. `company_id=None` on this helper means
personal scope, not an inherited company header. Preview/status helpers accept
an HTTP `timeout=30`. Server error JSON is returned unchanged.
When recovering an alias fork, use the receipt's resolved source UUID if known,
not `"-"`, for `source_conversation_id`; omit it if no receipt was received.

## More Documentation

Want to know more about AGiXT?  Check out our [documentation](https://josh-xt.github.io/AGiXT/) or [GitHub](https://github.com/Josh-XT/AGiXT) page.
