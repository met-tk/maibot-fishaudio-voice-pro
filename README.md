# Fish Audio TTS Voice Clone (Pro)

面向 MaiBot 的 Fish Audio 语音合成与即时音色复刻插件，严格依据 Fish Audio OpenAPI v1 `/v1/tts` 规范实现。

## 核心特性

1. **原生 Fish Audio OpenAPI v1 协议适配**
   - 使用标准请求头 `Authorization: Bearer <token>` 与 `model`（默认 `s2.1-pro-free`，支持切换 `s2.1-pro`、`s2-pro`、`drama-3`、`s1`）。
   - 同时支持 `application/json` 与基于 `msgpack` 的 `application/msgpack` 二进制参考音频上传。
   - 自动归一化接口地址：填写 `https://api.fish.audio`、`/v1` 或完整 `/v1/tts` 均可正确路由。
2. **多种音色获取策略（默认开箱即用）**
   - `auto`（推荐默认）：按 `云端 reference_id` -> `本地 voices/ 参考音频克隆` -> `模型默认音色` 顺序自动选取。
   - `reference_id`：使用 Fish Audio 平台公开或自建音色 ID（支持逗号分隔多音色）。
   - `clone`：通过 MessagePack 上传插件 `voices/` 目录中的本地参考音频（`.wav/.mp3/.flac/.ogg/.opus/.m4a`）及同名 `.txt` 台词文本完成即时零样本复刻。
   - `default`：不传参考音色，由模型自动匹配默认声线。
3. **高可靠与防串群设计**
   - 三段式超时保护（总超时 / 建连超时 / 读包静默超时）、429 与 5xx 退避重试，认证与余额不足错误不盲目重试。
   - 失败或并发队列繁忙时自动剥离 `[laughing]` / `（轻笑）` 等舞台标签后退回纯文本，不泄露内部提示词。
   - 严格解析当前会话 `stream_id`，多群并存且无法唯一确定目标流时不猜测跨群发送。
4. **云端与本地双重调音**
   - 支持服务端 `prosody`（语速、音量、响度归一化）及采样生成参数。
   - 可选本地 `ffmpeg` 滤镜链（变调、保持共振峰、EBU R128 响度归一化、高通滤波、裁剪静音等），默认全关零额外开销。
5. **多种语音触发策略与双滑块调节**
   - `[auto_voice]` 支持四种触发模式：`planner`（由 MaiBot Planner 根据语境自主决策是否发语音，推荐默认）、`probability`（纯随机概率模式）、`always`（全量出站文字自动转语音）、`off`（关闭自动与 Planner 主动触发）。
   - 提供两个独立的 WebUI 原生滑块：`planner_temperature`（Planner 语音温度，`0.0 - 1.0`）与 `random_probability`（纯随机概率，`0.0 - 1.0`），配合长文本按句分段发送及链接/代码块自动跳过。
6. **可控的输出语音存储与自动清理**
   - 新增 `[storage]` 配置段：默认 `save_output_audio = false`，音频全程在内存经 Base64 传输，不在磁盘保留历史语音文件；本地 `ffmpeg` 调音临时目录亦会在处理后立即删除。若需要将生成语音留档或排查，可开启 `save_output_audio = true` 指定 `output_dir`，并通过 `auto_delete`、`retain_minutes`（含设为 `0` 时发送后立即删除）与 `max_saved_files` 控制自动清理。
7. **情绪与语气参数自动注入（`[emotion]`）**
   - 默认开启（`enabled = true`），在发送给 Fish Audio 合成前自动按语句语义与起伏转折在句首及句中插入 `[开心的语气]`、`[平静]`、`[语气激动]`、`[哼歌]` 等行内标签（`s1` 模型自动转为圆括号格式）。
   - 支持三种模式：`hybrid`（默认：优先调用宿主大模型细腻标注，超时/失败无感回退本地规则）、`llm`（仅调用宿主大模型）与 `rule`（纯本地启发式规则零延迟注入）。
   - 内置原文防篡改校验与转换文本可视机制：控制台 `INFO` 日志及语音消息元信息/WebUI 观察面板默认展示实际带 `[情绪标签]` 的转换文本（`show_injected_text = true`），失败降级发文字时仍会自动剥离标签；如需在聊天窗口边听边看标签文本，可开启 `send_injected_text_to_chat = true`。

## 目录结构

```text
custom_maibot-fishaudio-voice-pro/
├── _manifest.json
├── __init__.py
├── plugin.py              # MaiBot SDK 插件入口、配置模型、Tool 与公共 API
├── tts_service.py         # Fish Audio /v1/tts 异步 HTTP / MessagePack 客户端
├── audio_processor.py     # 本地 ffmpeg 调音链与裸 PCM -> WAV 封装
├── config.example.toml    # 配置模板
├── config.toml            # 当前生效配置
├── requirements.txt       # 依赖声明（aiohttp>=3.8.0, msgpack>=1.0.0）
├── voices/                # 可选：放置本地克隆参考音频与同名 .txt 台词文本
└── tests/
    └── test_plugin.py     # 离线单元测试
```

## 快速配置

在 `config.toml` 中填写 `[voice]` 段的 `fish_api_key` 与 `model`（默认 `s2.1-pro-free`）：

```toml
[voice]
fish_api_key = "your-fish-audio-api-key"
api_base_url = "https://api.fish.audio/v1/tts"
model = "s2.1-pro-free"
voice_mode = "auto"
reference_id = ""
audio_format = "mp3"
```

### 使用本地音色克隆（可选）

1. 在插件目录新建 `voices/` 文件夹。
2. 放入 5-15 秒清晰干声参考音频，例如 `voices/default.wav`（或 `.mp3`）。
3. 推荐同时创建同名文本文件 `voices/default.txt`，填入参考音频中对应的准确文字台词，以提升 Fish Audio 克隆还原度。
4. 将 `[voice].reference_id` 留空且保持 `voice_mode = "auto"`（或设为 `"clone"`），插件将自动使用 `application/msgpack` 编码发送参考音色。

### 情绪与语气自动注入（`[emotion]`）

插件默认开启行内情绪注入，也可以在 `config.toml` 或 MaiBot WebUI 中按需开关或微调：

```toml
[emotion]
enabled = true
mode = "hybrid"          # hybrid（LLM 优先 + 规则兜底） | llm | rule
model_task = ""          # 留空使用宿主默认模型任务，也可填 utils / replyer 等
temperature = 0.3
timeout_seconds = 8.0
tag_presets = "开心的语气, 平静, 语气激动, 温柔, 撒娇, 哼歌, 小声说话, 叹气, 轻笑, 疑惑, 委屈, 严肃, 俏皮, 惊讶"
custom_instruction = ""
skip_if_already_tagged = true
show_injected_text = true
send_injected_text_to_chat = false
```

开启后，例如输入原始台词：

`嘿嘿，快看这里！如果你现在还没关注臭企鹅，那你的肚子肯定会变得圆滚滚的！`

插件会在合成前自动注入为：

`[开心的语气]嘿嘿，快看这里！[平静]如果你现在还没关注臭企鹅，那你的肚子肯定会变得圆滚滚的！`

并默认在插件/TTS `INFO` 日志及 WebUI 观察面板的语音文本摘要中直接展示注入后的完整内容（语音合成失败降级为纯文本时，仍会自动剥离标签仅发干净原文）；若需要直接在 QQ/微信聊天窗口查看实际转换的带标签文本，可将 `send_injected_text_to_chat` 设为 `true`。

## 暴露组件

- `@Command("fishaudio")`：提供聊天强制语音转换命令：
  - `/fishaudio <文本>`：按当前 `[emotion].enabled` 设置项决定是否注入情绪标签，并强制合成为语音发送到当前会话（不受 `[auto_voice].mode` 限制）。
  - `/fishaudio emo <文本>`：无论 `[emotion].enabled` 是否开启，均强制执行情绪与语气自动注入后再合成为语音发送。
- `@ReplyExtension("voice_reply")`：融入 MaiBot 原生 `reply` / `reply_supplement` 流程，在 `mode = "planner"` 时由 Planner 自行决定是否选用语音并可附带情绪风格提示；合成失败时自动无损保留原文字。
- `@HookHandler("maisaka.planner.before_request")`（`fish_planner_voice_policy`）：在 `planner` 模式下按 `planner_temperature` 动态注入语音决策引导；在非 `planner` 模式下自动隐藏语音扩展与直发工具，避免误触发。
- `@HookHandler("send_service.before_send")`（`fish_auto_voice`）：在 `mode = "always"` 或 `"probability"` 时按策略将出站纯文本按句/长度分段转为 `voice` 消息发送，失败自动放行纯文本；同时清理语音消息上不兼容的引用标记。
- `@Tool("send_voice_reply")`（`visibility="visible"`）：供 MaiBot 对话决策直接显式调用的语音回复工具（参数：`text` 必填，`voice` / `style` 可选）。
- `@API("fish_tts")`：供其他插件跨插件调用的公共合成接口，返回 `{success, audio_base64, audio_bytes_len, format, model, voice, text}`（开启 `[storage].save_output_audio` 且 `retain_minutes > 0` 时附带 `audio_path`）。


## 安装与部署

1. 克隆仓库至 MaiBot 的 `plugins` 目录：
   ```bash
   git clone https://github.com/met-tk/maibot-fishaudio-voice-pro.git
   ```
2. 在 MaiBot 所使用的 Python 环境中安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 复制示例配置并填入你的 Fish Audio API Key：
   ```bash
   copy config.example.toml config.toml
   ```
4. 重启或重载 MaiBot 插件即可生效。若需使用本地 `ffmpeg` 调音功能，请确保系统 `PATH` 中已安装 `ffmpeg`（或在 `config.toml` 中指定 `ffmpeg_path`）。

## 开源协议

本项目基于 [MIT License](LICENSE) 开源。
