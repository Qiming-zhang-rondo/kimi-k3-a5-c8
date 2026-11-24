# DSH 0.2.2-rc.1 push notification transport

This is a source-verified protocol note, not a claim that the user's DSH instance has been contacted. Target: official `deepseek-ai/deepseek-harness`, tag `dsh-v0.2.2-rc.1`, commit `66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db`. Do not substitute current main or a headless SDK process.

## Existing-session admission

Send `POST /api/session/prompt` to the existing Host, with its authenticated Cookie and JSON:

```json
{
  "type": "client-request",
  "rpcId": "HTTP-request-UUID",
  "method": "session/prompt",
  "payload": {
    "args": {
      "request": {
        "requestId": "stable-notification-UUID",
        "sessionId": "exact-existing-session-id",
        "mode": "queue",
        "content": [{"type": "text", "text": "trusted notification text"}],
        "clientTimeZone": "Asia/Shanghai"
      }
    }
  }
}
```

`clientTimeZone` is optional; this release does not call it `timeZone`. The correlation `rpcId` and durable `requestId` have separate purposes. Preserve the same `requestId` when retrying an ambiguous notification after timeout. Use a deterministic identity scoped to repository, branch, target session and accepted push SHA; do not interpolate commit messages into the trusted prompt.

Accept success only when HTTP succeeds and the response is `type: "server-response"`, the same `rpcId`, `result.ok === true`, and `result.value.accepted === true`. HTTP 200 alone can carry a business error. Acceptance means queued, not completed.

Sources: [request and response schema](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/client/connection/src/rpc-schema.ts#L34), [HTTP dispatch](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/client/connection/src/rpc-host.ts#L310), [exact args envelope](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/api/gateway/src/index.ts#L1261), [prompt fields and accepted receipt](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/api/session-controller/src/types.ts#L321).

The controller resolves a live session or resumes that exact persisted ordinary session. Missing sessions return `session/not-found`; subagent ownership is rejected; another writer can return `session/writer-held`. It does not fall back to creating a session. Mode `queue` calls `agent.followup`; it does not steer/cancel an active turn. Already queued or logged `requestId` values return the original acceptance.

Sources: [resolve/resume](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/api/session-controller/src/agent.ts#L174), [missing persisted session and resume-only path](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/api/session-controller/src/agent.ts#L294), [queue and dedup](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/api/session-controller/src/commands.ts#L304), [durable identity lookup](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/api/session-controller/src/commands.ts#L542).

## Authentication and lifecycle

The explicit API-client flow is:

1. Read a user-configured launch token from a private file. Do not discover credentials by scraping `.credentials.yaml`, browser stores, or private IPC.
2. `GET /connect?token=<url-encoded-token>&interactions=none`, with automatic redirects disabled.
3. Require 302, parse `Set-Cookie`, and resolve `Location` against that request URL. This release returns `./api/remote.mux`. Validate that it stays on the configured authority. Convert only the scheme `http` to `ws` or `https` to `wss`.
4. Open that lifecycle WebSocket using the same Cookie. No subprotocol and no logical-stream messages are required. Do not subscribe to `$events` or claim human approval interactions.
5. Make the HTTP prompt request, check the accepted receipt, then close the socket in `finally`. Its closure revokes this API Cookie and releases its Peer.

The Host uses Ping frames while the socket is open. A short-lived `websocket-client` connection still needs a receive pump to answer Ping during a slow HTTP admission; an unused `create_connection()` object does not process incoming control frames by itself. This is transport maintenance, not GitHub polling. A read-only state request can use the same lifecycle without sending a model message.

Sources: [official migration instructions](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/docs/upgrade-guide/v0.2.1-alpha.2/operator-connect/guide.md#L14), [302 and Cookie creation](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/client/connection/src/browser-auth.ts#L314), [single-socket ownership and release](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/client/connection/src/rpc-host.ts#L144), [upgrade cleanup](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/api/gateway/src/index.ts#L273), [WebSocket setup](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/api/gateway/src/stream-server.ts#L69).

An explicitly supplied existing **browser Cookie** also works with HTTP-only calls and uses the shared owner Peer; it creates no API Peer per push. Do not treat an API Cookie as an interchangeable durable browser Cookie. Repeating `/connect` without ever attaching a lifecycle socket leaves unused Peers until Host shutdown. Avoid that implementation. Token exchange accepts no Authorization-header bearer token. Keep token, Cookie, token-bearing URLs, and raw auth errors out of logs.

Source: [admission distinguishes browser and API Cookies](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/client/connection/src/rpc-host.ts#L128), [authentication contract and limitations](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/client/connection/README.md#L38).

## No-message check

Use the same envelope with `method: "session/state"`, URL `/api/session/state`, and `payload: {"args":{"request":{"sessionId":"exact-id"}}}`. This is a passive read; an inactive persisted session can return `active: false`, `running: false`, `queue: null` without activating an Agent. A valid response proves authentication, route compatibility and target lookup; it does not prove notification receipt or future model execution.

Sources: [state endpoint](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/api/session-controller/src/index.ts#L566), [passive implementation](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/api/session-controller/src/state/controller.ts#L134), [state fields](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/api/session-controller/src/types.ts#L606).

To discover IDs without sending a prompt, use `/api/session/list` with `method: "session/list"` and **`payload: {"args":{"_request":{}}}`**. This release's method parameter is `_request`, not `request`; Gateway validates exact named arguments. The result is `{items: SessionSummary[]}`. Each summary has `sessionId`, `agentAvailable`, `running`, `blank`, `updatedAt`, and optional `cwd`, `parentSessionId`, `origin`, and `projections`. There is no separate top-level `title` field. The operation lists visible sessions without resuming an Agent. Do not use ACP's similarly named `session/list` protocol as a substitute.

Sources: [list method and exact parameter name](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/api/session-controller/src/index.ts#L272), [list request/result](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/api/session-controller/src/types.ts#L244), [summary fields](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/api/session-controller/src/types.ts#L151).

No dedicated runtime-version HTTP endpoint was verified in this investigation. Do not invent `/api/version` or report a successful state request as exact version attestation. The config can declare `0.2.2-rc.1`; actual version evidence must come from the running installation. An unrelated `dsh --version` on PATH need not identify the desktop Host. The pinned desktop Host constructs its own authenticated loopback URL and passes it to its parent; this note does not authorize scraping that private IPC.

Source: [desktop Host startup URL](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/apps/desktop-host/src/index.ts#L142).

Verified official CLI spelling for this release: `dsh --version`, `dsh web --help`, `dsh web`, and `dsh --profile web --port 8080`. Starting a Web profile is not an automatic attachment to an existing TUI/desktop process; use the actual Host serving the target. Sources: [CLI args](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/apps/cli/src/args.ts#L89), [CLI examples](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/apps/cli/README.md#L25).

The SDK's `DeepSeekHarness` starts its own runtime subprocess. It is not a substitute for addressing the already-running desktop/Web Host and can conflict with a session's existing writer.

Source: [SDK process ownership](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/packages/sdk/client/src/api.ts#L14).
