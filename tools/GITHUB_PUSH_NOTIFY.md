# 通用 GitHub push → Codex 聊天接收器

只复制 `github_push_notify.py` 即可跨项目使用：Python 3.10+ 标准库，无 pip/NPU 依赖。
一份配置绑定一个仓库、分支和既有聊天；多个项目运行多个实例，各用自己的端口和配置目录。
工作流是 GitHub push → HTTPS `/github` → 验签/过滤/去重 → 官方 `codex queue` → 既有聊天。
没有定时任务，也不轮询 GitHub。

## 先说明验证范围

2026-11-24 已在本机实测官方 `codex queue --thread UUID --message TEXT`：测试消息实际
进入了当前聊天。因此本地模式**不要求**另开 app-server 或手填 `--remote`。
之前根据默认 daemon socket 不存在判断消息入口不可用，是不成立的；现已修正适配。

本地自动化测试覆盖签名 HTTP 请求到真实子进程适配器、目标/参数、隔离去重和失败重投。
测试中的 Codex 子进程是可执行测试替身，不启动推理。真实聊天入队测试与这些测试分开记录。
GitHub 公网回调注册仍是每个部署的一次性配置；只有实际 delivery 成功且指定聊天收到事件，
才说明该部署端到端接通。`check` 或 ping 成功不能代替这一项。

## 1. 初始化每个项目的配置

在运行目标 Codex 聊天的那台电脑上执行，不是在待测 A5 容器中安装 Codex。
确保 `codex queue --help` 可用；UUID 使用目标聊天的真实 ID。在聊天中可让助手读取
当前会话的 `CODEX_THREAD_ID`，不要把项目目录名当作聊天 ID。

```bash
python3 tools/github_push_notify.py init \
  --dir "$HOME/.config/github-push-notify/my-project" \
  --repo OWNER/REPO \
  --branch main \
  --thread YOUR_EXISTING_CHAT_UUID
```

此命令生成 `config.json`、无末尾换行的 `config.secret` 并配置私有状态目录；serve时创建状态目录。
已有配置或密钥不会被覆盖。
配置/密钥应放仓库外。密钥文件权限600，内容不打印到终端、不提交GitHub。
若 `codex` 不在 PATH，加 `--codex /absolute/path/to/codex`。

本项目可用 `--repo Qiming-zhang-rondo/kimi-k3-a5-c8 --branch sfa-cold-merge`，
并加 `--ignore-commit-prefix '[sfa-review]'`；通用默认忽略前缀为 `[codex-review]`。
前缀仅用于防止助手自己的审查提交再次触发；若同一push还含其他提交，仍会通知。

## 2. 检查配置并启动

```bash
python3 tools/github_push_notify.py check --config "$HOME/.config/github-push-notify/my-project/config.json"
python3 tools/github_push_notify.py serve --config "$HOME/.config/github-push-notify/my-project/config.json"
```

`check` 读取配置、密钥及CLI能力，调用 `codex queue --help`；不发送模型消息。
`serve` 监听 `127.0.0.1:8765`，仅事件到达才工作。改端口可在init时加 `--port 8766`。
接收器应持续运行；停止它后不再接收事件。不要为保持运行另加定时轮询。

## 3. 测试接收器存活与签名

另开终端运行：

```bash
python3 tools/github_push_notify.py test --config "$HOME/.config/github-push-notify/my-project/config.json"
python3 tools/github_push_notify.py status --config "$HOME/.config/github-push-notify/my-project/config.json"
```

`test` 只发送带正确签名的 ping，不发送push、不会唤醒模型；可用 `--url https://YOUR_HOST/github`
检查部署后的HTTPS路由。`status` 只读本地投递状态，不访问GitHub。
注意 `pong` 只证明请求/验签可达，不代表仓库Webhook已配置，也不代表聊天已收到消息。

## 4. 配置 GitHub 的实际 push 回调

需要一个能到达本机 `/github` 的HTTPS地址，以及该仓库Webhook管理权限。
使用已有反向代理/隧道即可。例如已安装cloudflared时，可临时开发验证：

```bash
cloudflared tunnel --url http://127.0.0.1:8765
```

取生成的HTTPS地址，末尾加 `/github`。临时地址重启会变，稳定使用请配置固定HTTPS入口。
回调电脑、接收器及通道须在线。

在仓库 Settings → Webhooks → Add webhook 填写：

| 项目 | 值 |
| --- | --- |
| Payload URL | `https://YOUR_HOST/github` |
| Content type | `application/json` |
| Secret | 本机 `config.secret` 的精确文本，不要附换行 |
| Events | 仅 `push` |
| SSL verification | 开启 |

保存后GitHub会发ping；再推送一次目标分支或在Recent Deliveries手动重新投递合法push。
验收必须同时看到delivery成功、接收器状态为已入队、指定聊天收到相同SHA。
若聊天正在工作，消息可能等待当前响应结束再处理。入队成功不等于审查已经完成。
本工具不使用额外OpenAI API key，复用本机已登录的官方Codex CLI。

## 跨项目适配

`github_push_notify.example.json` 展示全部主要字段；用init生成自己的配置，不直接使用示例UUID。
相对 `secret_file`、`state_dir` 路径相对于配置文件目录；本机默认 `codex_remote` 为null。
可选 `codex_remote` 仅接受本机Unix socket或loopback WebSocket；远端app-server需先通过
已有SSH转发等方式映射到本机。仅使用确实承载该聊天的已有app-server，不能随便新开daemon
再声称通知到了原聊天。脚本不会自行创建聊天、启动daemon或使用 `exec resume`。

`message_template` 是可信本地配置，支持 `{repo}`、`{branch}`、`{sha}`；禁止读取payload里的
commit message作为执行指令。建议模板只要求审查已绑定项目的新增代码/结果。
同一个SHA在不同仓库、分支或聊天里分别去重，不会串项目。更换绑定推荐新建配置目录。

## 失败怎么看

| HTTP/状态 | 意义与处理 |
| --- | --- |
| 200 pong | 验签ping成功，不触发模型 |
| 200 ignored_* | 非目标repo/ref、删除事件或自审查提交，正常过滤 |
| 200 duplicate | 该路由的delivery或SHA已成功入队 |
| 202 queued | 官方CLI已接受入队；仍需看指定聊天是否收到 |
| 400/401/413 | 格式/签名/大小不符合合同；检查配置和原始请求 |
| 503 queue_failed | CLI路径、登录、目标聊天或可选远程连接有问题；本次不记成功去重 |

GitHub**不会自动重投失败delivery**。修复后到Recent Deliveries手动Redeliver，脚本允许失败的
delivery再次尝试；不增加定时重试。CLI调用有短超时，避免占满GitHub的10秒响应窗口。
进程在“CLI已接受但成功状态未写入”之间崩溃时，人工重投可能重复通知，不能宣称严格exactly-once。
默认请求上限1MiB，仅loopback监听，HMAC使用常量时间比较，日志不输出密钥/原始payload。

## 维护与测试

```bash
python3 -m unittest discover -s tests -p test_github_push_notify.py
```

本工具只负责把事件送入指定聊天；项目的代码权限、审查方式和后续工作仍由该聊天原有指令控制。
首次部署后保留实际delivery状态与聊天回执，避免只看本地测试就宣布部署完成。

参考：[官方Codex queue](https://learn.chatgpt.com/docs/developer-commands#codex-queue)、
[GitHub验签](https://docs.github.com/en/webhooks/using-webhooks/validating-webhook-deliveries)、
[10秒响应约束](https://docs.github.com/en/webhooks/using-webhooks/best-practices-for-using-webhooks)、
[失败不自动重投](https://docs.github.com/en/webhooks/using-webhooks/handling-failed-webhook-deliveries)、
[临时HTTPS回调](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)。
