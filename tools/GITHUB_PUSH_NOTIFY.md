# GitHub push 事件通知：无定时器、无分支轮询

用户要求事件驱动，误建的heartbeat自动任务已经删除。
本脚本在收到已验证的GitHub Webhook后才调用现有Codex聊天的消息入队接口。
它不运行模型审查、不访问NPU，也不自行启动另一个Codex daemon/chat。

## 当前接通状态

- 接收器和本地模拟测试随代码交付。
- 本机CLI确实支持 `codex queue --thread ID --message TEXT --remote ENDPOINT`。
- 当前桌面聊天尚未找到可用的app-server消息endpoint；默认control socket不存在。
  因此不能声称当前已能被GitHub push自动唤醒。
- 当前GitHub连接提供仓库文件/提交能力，不提供管理Webhook的工具。
  仓库Webhook尚未注册，公网HTTPS回调尚未部署；不创建定时器来代替。

要实际接通，需要一个**确实承载目标聊天的**可访问app-server endpoint，以及仓库
push Webhook到本机接收器的HTTPS路由。不能随便新开daemon再宣称通知到了当前聊天。

## 使用已有可用入口时

在Mac本机、原有chat所属的app-server环境中运行；不是在A5容器中安装Codex。
参数中的ID和endpoint必须先验证真实存在：

```bash
python3 tools/github_push_notify.py \
  --thread EXISTING_THREAD_ID \
  --codex /absolute/path/to/codex \
  --codex-remote unix:///absolute/path/to/current-app-server.sock \
  --secret-file /private/path/to/github-webhook-secret \
  --state-dir /private/path/to/github-webhook-state \
  --port 8976
```

密钥由本机生成并放在仓库外，至少32字节，文件权限600，无末尾换行。
脚本使用文件的精确字节，不自动去除空白；不要把密钥发到聊天或提交GitHub。
将接收器暴露为受控HTTPS Webhook地址后，由有仓库Webhook管理权限的操作者配置：
Content type为application/json、仅push事件、Secret为相同密钥。接收器POST路径为 `/github`。
确认GitHub delivery成功且目标聊天确实收到同一commit事件后，才记为CONNECTED。
公网路由或本机app停止时没有自动运行保证；记录实际连接状态。

## 行为及验证

验签使用原始body和X-Hub-Signature-256，常量时间比较；仅接受指定仓库/分支，校验delivery
和commit标识，持久化去重。消息仅包含验证后的repo/ref/SHA，不把commit message等外部文本
作为指令，也不执行payload中的shell。queue失败不得写入成功去重记录，供GitHub重新投递。
`[sfa-review]`前缀的自审查提交跳过，防止审查自己的回推造成循环。

本地测试只用模拟queue：验签、分支过滤、重复投递、queue失败及HTTP请求都不得启动真实推理。
确认目标聊天消息通路时才做一次有标记的实测，不能把本地模拟测试称为Webhook端到端成功。

参考：[GitHub验签合同](https://docs.github.com/en/webhooks/using-webhooks/validating-webhook-deliveries)、
[Codex官方消息队列说明](https://developers.openai.com/blog/mastering-codex-remote-for-engineering)。
