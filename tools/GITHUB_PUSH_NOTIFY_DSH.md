# GitHub push → 已有 DSH 会话

适配 **DeepSeek Harness 0.2.2-rc.1**，接口核对自官方同版本源码
[`66491b05`](https://github.com/deepseek-ai/deepseek-harness/tree/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db)。
接收器收到 GitHub push 后验签、检查仓库/分支并去重，再向指定已有 DSH 会话排队。
没有 GitHub 轮询、定时任务，也不启动新模型会话。

本机没有安装 DSH，尚未对真实 DSH 验证收件。协议核验、本地测试、接收器 ping 与真实
端到端投递是不同验证层次；只有目标 DSH 会话收到正常 push 的相同 SHA，才算该部署接通。

## 准备目标 DSH Host

接收器需连接**正在运行且能访问目标会话的 DSH Web Host**。准备：

- 该 Host 的完整启动页 URL，包含 `?token=...`；只保存到仓库外的私有文本文件。
- 已有会话的准确 ID；不使用标题、目录名或 Codex UUID 代替。
- Python 3.10+，以及本工具指定的 `websocket-client==1.9.0`。

沿用已有 DSH 安装，不需要重装。官方 npm 包名是 `@deepseek-ai/dsh`，本次核对版本为
`0.2.2-rc.1`；接收器本身不运行 Node 或启动 DSH。

运行中桌面/Web Host 的真实 URL 应由该实例的正常入口取得。不要读取 `.credentials.yaml`
或私有 IPC 来提取凭据，不把启动 URL、Cookie 发到聊天或提交 Git。

仅有 `dsh` CLI/TUI 并不代表存在可附加的 HTTP Host。如果确实尚未开启 Web，官方此版本支持：

```bash
dsh --version
dsh web --help
dsh web
```

也支持 `dsh --profile web --port 8080`；使用它打印的实际带 token URL。这些命令的语法来自
[官方 CLI 文档](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/apps/cli/README.md#L9)
和 [参数实现](https://github.com/deepseek-ai/deepseek-harness/blob/66491b05e3f05ad35bcdd4f6cd7ed26041d9e3db/apps/cli/src/args.ts#L89)。
不要为了通知已运行的桌面对话另开 headless/SDK 进程；若目标已被另一个 Host/TUI 占用，
应连接其所属 Host，不能复制会话或强占写锁。

## 安装与配置：Unix

从仓库根目录执行，或把以下工具放到同一目录：`github_push_notify_dsh.py`、
`github_push_notify.py`、`requirements-dsh-notify.txt`。

```bash
python3 -m pip install -r tools/requirements-dsh-notify.txt
mkdir -p "$HOME/.config/github-push-notify"
```

用文本编辑器在 `$HOME/.config/github-push-notify/dsh-launch-url.txt` 保存一行完整的
DSH 启动页 URL。不要将 token 写在命令参数中。保存后：

```bash
chmod 600 "$HOME/.config/github-push-notify/dsh-launch-url.txt"
python3 tools/github_push_notify_dsh.py sessions --connect-url-file "$HOME/.config/github-push-notify/dsh-launch-url.txt"
python3 tools/github_push_notify_dsh.py init \
  --dir "$HOME/.config/github-push-notify/my-dsh-project" \
  --repo OWNER/REPO --branch main \
  --session EXACT_EXISTING_SESSION_ID \
  --connect-url-file "$HOME/.config/github-push-notify/dsh-launch-url.txt"
python3 tools/github_push_notify_dsh.py check --config "$HOME/.config/github-push-notify/my-dsh-project/config.json"
python3 tools/github_push_notify_dsh.py serve --config "$HOME/.config/github-push-notify/my-dsh-project/config.json"
```

`sessions` 只读列出该 Host 可见的会话，不生成回答。按 ID、工作目录、运行状态辨认目标，
不要根据列表顺序自动选第一个。该版响应没有单独的顶层 `title` 字段。

## 安装与配置：Windows PowerShell

从仓库根目录执行。创建仓库外的私有目录，用记事本保存启动 URL；不要把 URL 粘到终端命令。

```powershell
py -3 -m pip install -r tools\requirements-dsh-notify.txt
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\GitHubPushNotify" | Out-Null
notepad "$env:LOCALAPPDATA\GitHubPushNotify\dsh-launch-url.txt"
```

保存文本后继续：

```powershell
py -3 tools\github_push_notify_dsh.py sessions --connect-url-file "$env:LOCALAPPDATA\GitHubPushNotify\dsh-launch-url.txt"
py -3 tools\github_push_notify_dsh.py init `
  --dir "$env:LOCALAPPDATA\GitHubPushNotify\my-dsh-project" `
  --repo OWNER/REPO --branch main `
  --session EXACT_EXISTING_SESSION_ID `
  --connect-url-file "$env:LOCALAPPDATA\GitHubPushNotify\dsh-launch-url.txt"
py -3 tools\github_push_notify_dsh.py check --config "$env:LOCALAPPDATA\GitHubPushNotify\my-dsh-project\config.json"
py -3 tools\github_push_notify_dsh.py serve --config "$env:LOCALAPPDATA\GitHubPushNotify\my-dsh-project\config.json"
```

Windows 目录应由当前用户的文件权限保护；POSIX `0600` 不等于 Windows 的完整 ACL 设置。
不要放共享盘或其他普通用户可读的目录。

## init、check 和 test 各做什么

`init` 生成仓库外的 `config.json`、独立 GitHub 验签密钥 `config.secret`，以及私有
`config.connect-url`。最后一个文件是读取后规范化保存的 DSH token URL；原输入文件不改。
已有配置/密钥不覆盖。默认监听 `127.0.0.1:8766`，可在 init 加 `--port`。

启动 URL 可为根路径、带挂载前缀且以 `/` 结尾的路径，或 `/connect`，均须只包含一个
`token` 查询参数。例如挂载前缀会保留到认证、WebSocket和RPC路径，不被丢弃。
不接受 URL 用户名/密码、fragment 或其他查询参数。凭据文件和配置允许 UTF-8 BOM，
兼容 Windows 文本编辑器；生成的文件使用普通 UTF-8。

`check` 验证配置并向 DSH 发送只读 `session/state`，不发送 prompt。它使用短生命周期
WebSocket 管理 API Cookie/Peer，结束时关闭；这不是后台轮询。成功仅证明认证、目标查询
和当前接口可用，不证明已发送通知。配置中的 `dsh_version` 也不是服务端版本检测结果。

`serve` 打开 Webhook 监听端口前，也执行同样的只读检查；目标 Host/会话检查失败就退出。
返回 `CONFIG_READY` 时，`runtime_version_verified`、`chat_delivery_tested` 和
`model_message_sent` 仍为 false。

另开终端可检查接收器和本地记录：

```bash
python3 tools/github_push_notify_dsh.py test --config /PATH/TO/config.json
python3 tools/github_push_notify_dsh.py status --config /PATH/TO/config.json
```

Windows 将 `python3` 换成 `py -3`，使用实际配置路径。`test` 只发签名 ping，不给 DSH
发消息；可加 `--url https://YOUR_HOST/github` 检查外部 HTTPS 路由。`status` 只读本地数据库。

## GitHub 一次性接入

保持接收器和 DSH Host 运行，将 HTTPS `/github` 转发至 `127.0.0.1:8766/github`。
不要把 DSH 自己的 token URL 或 API 端口当成 GitHub Payload URL。

在仓库 Settings → Webhooks → Add webhook 中设置：

| 字段 | 内容 |
| --- | --- |
| Payload URL | 接收器的 `https://YOUR_HOST/github` |
| Content type | `application/json` |
| Secret | `config.secret` 的精确内容，无附加换行；不是 DSH token |
| Events | 仅 `push` |
| SSL verification | 开启 |

HTTPS 入口、临时隧道及 GitHub 配置步骤也见
[通用接收器说明](GITHUB_PUSH_NOTIFY.md#4-配置-github-的实际-push-回调)，端口改为本实例的8766。
配置需要仓库 Webhook 管理权限，但不需要将 GitHub token 交给这个接收器。

最终用一次正常目标分支 push，或对已有合法 push 手动 Redeliver，检查三项：GitHub
delivery 成功、本地记录已入队、**指定 DSH 会话出现相同 SHA**。若会话正在工作，通知
按 `queue` 排到后续；接受入队不代表审查完成。这里不提供会消耗模型调用的“测试 prompt”。

## 维护与故障处理

- DSH Host 重启后 launch token 可能改变。使用新启动 URL 更新配置旁的
  `config.connect-url` 私有文件，再执行 check；更换 Host 的地址/端口时也需更新
  `dsh_base_url`。不要把“只改配置版本”当作跨版本适配。
- `session/not-found` 表示 ID 或 Host 不对应；`session/writer-held` 表示会话被另一运行时
  占用。接收器不新建会话、不绕过写锁、不修改会话日志。
- 401/403 通常属于过期凭据或来源校验；不要关闭鉴权。HTTP 200 也可能包含 DSH
  `result.ok=false`，脚本需以 `accepted:true` 为最终入队回执。
- 通知失败返回503，不写入成功去重记录。GitHub 不自动重投失败 delivery；修复后在
  Recent Deliveries 手动 Redeliver。重复事件使用相同持久 `requestId`，DSH按它检查
  已接受的输入；仍不承诺数据库/Host跨进程崩溃下严格 exactly-once。
- 接收器、HTTPS 通道或 DSH 停止后无法投递。保持进程在线即可，不要另设GitHub轮询或定时器。

示例配置见 [github_push_notify_dsh.example.json](github_push_notify_dsh.example.json)。
一份配置绑定一个仓库/分支/已有会话，多个项目使用独立配置目录、端口和密钥。
可信 `message_template` 只支持 `{repo}`、`{branch}`、`{sha}`；提交消息不会进入指令。
自审查提交前缀通过 `ignore_commit_prefixes` 配置，本项目可设为 `[sfa-review]`。

协议、鉴权生命周期、只读查询及官方源码位置见
[DSH_022_RC1_PROTOCOL.md](DSH_022_RC1_PROTOCOL.md)。本工具只是投递器；会话原有权限和任务
指令继续控制后续工作，不将普通仓库文本提升为自动执行授权。
