"""Fish Audio TTS 与音色克隆 MaiBot 插件。

提供基于 Fish Audio OpenAPI v1 (POST /v1/tts) 的语音合成与即时音色复刻能力，
支持 s2.1-pro-free / s2.1-pro / s2-pro / drama-3 / s1 模型，支持云端 reference_id
以及本地 voices/ 目录参考音频 MessagePack 即时克隆，并在失败时自动降级为纯文本回复。
"""

from __future__ import annotations

import asyncio
import base64
import copy
import difflib
import logging
import random
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import maibot_sdk
from maibot_sdk import (
    API,
    CONFIG_RELOAD_SCOPE_SELF,
    Command,
    Field,
    HookHandler,
    MaiBotPlugin,
    PluginConfigBase,
    Tool,
)
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder, ToolParameterInfo, ToolParamType
from pydantic import model_validator

ReplyExtension = getattr(maibot_sdk, "ReplyExtension", None)
if ReplyExtension is None:  # pragma: no cover - 兼容旧版 SDK 降级占位
    def ReplyExtension(*_args: Any, **_kwargs: Any):  # type: ignore[no-redef]
        def _noop_decorator(func: Any) -> Any:
            return func

        return _noop_decorator

from .audio_processor import (
    DEFAULT_PCM_SAMPLE_RATE,
    RAW_PCM_FORMATS,
    AudioProcessor,
    build_audio_processor,
    wrap_raw_pcm_to_wav,
)
from .tts_service import (
    FishAudioTTSService,
    FishSynthesisOptions,
    ReferenceSample,
)

PLUGIN_DIR = Path(__file__).resolve().parent
_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a")
_MAX_REFERENCE_AUDIO_BYTES = 25 * 1024 * 1024
_MAX_REFERENCE_TEXT_BYTES = 64 * 1024
_NON_BODY_SEGMENTS = frozenset({"at", "reply", "face"})
_MEDIA_DICT_TYPES = frozenset({"voiceurl", "imageurl", "videourl", "record", "image", "video", "file", "music"})
_FALLBACK_BYPASS_TTL_SECONDS = 15.0
_OUTPUT_FILE_NAME_RE = re.compile(r"^fishaudio_\d{8}_\d{6}_[0-9a-f]{8}\.(mp3|wav|opus|pcm)$")
_VOICE_REPLY_EXTENSION_NAME = "voice_reply"
_DEFAULT_VOICE_EXTENSION_FULL_NAME = f"custom.maibot-fishaudio-voice-pro.{_VOICE_REPLY_EXTENSION_NAME}"
_PLANNER_VOICE_ITEM_PREFIX = "fishaudio-voice-planner-"
_AUTO_VOICE_MODES = frozenset({"planner", "probability", "always", "off"})
_UNSUITABLE_TEXT_PATTERNS = ("http://", "https://", "```", "![", "|--")
_PLANNER_TEMPERATURE_TIERS: tuple[tuple[float, str, str], ...] = (
    (
        0.0,
        "极度克制（仅明确要求）",
        "仅当用户明确要求发送语音（例如“发条语音/念给我听”）或对方刚发来语音且适合用语音回敬时，才启用语音扩展并设置 explicit_request=true；其余所有情况一律保持纯文字回复。",
    ),
    (
        0.3,
        "低频克制",
        "日常回复默认以文字为主；仅在撒娇、安慰情绪、道晚安、表达想念、简短接梗吐槽或回应对方语音等高情绪契合场景才启用语音扩展，并如实评估 suitability（0.0-1.0）。",
    ),
    (
        0.65,
        "均衡自然",
        "结合语境与表达自然度自行权衡：口语感强、带情绪互动或简短问候的回复可积极启用语音扩展；涉及长段说明、步骤教程、代码、链接或客观资料检索时请坚持用文字回复。",
    ),
    (
        0.95,
        "高频倾诉",
        "只要回复内容适合口语朗读（篇幅适中、非代码/网址/列表/长篇科普），请优先启用语音扩展发送语音；仅偏书面或结构化信息保留纯文字。",
    ),
    (
        1.0,
        "极高倾向",
        "除明显不适合语音朗读的内容（长文、代码块、链接、数据表格）外，积极主动选用语音扩展替代纯文字回复。",
    ),
)

_STYLE_BLOCK_RE = re.compile(r"<style>.*?</style>", re.IGNORECASE | re.DOTALL)
_SQUARE_TAG_RE = re.compile(r"\[[^\]\r\n]{1,48}\]")
_CN_PAREN_TAG_RE = re.compile(r"（[^（）\r\n]{1,28}）")
_EN_PAREN_TAG_RE = re.compile(r"\((?!\d+\))[^()\r\n]{1,36}\)")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_SPEAKABLE_CHAR_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_SENTENCE_SPLIT_RE = re.compile(
    r"(?:(?<=[。！？；!?])(?![”’\"'」』）)])\s*|(?<!\d)(?<=\.)(?=\s|$)\s*|(?<=[。！？；!?.][”’\"'」』）)])(?:\s+|(?=[“‘\"「『])))"
)
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[，、：,:])\s*")


class PluginSectionConfig(PluginConfigBase):
    """插件基础开关配置。"""

    __ui_label__ = "插件设置"

    enabled: bool = Field(
        default=True,
        description="是否启用 Fish Audio 语音插件",
        json_schema_extra={"label": "启用插件", "order": 1},
    )
    config_version: str = Field(
        default="1.2.0",
        description="配置文件版本号",
        json_schema_extra={"label": "配置版本", "order": 2},
    )


class VoiceSectionConfig(PluginConfigBase):
    """Fish Audio 接口与音色配置。"""

    __ui_label__ = "音色与接口配置"

    fish_api_key: str = Field(
        default="",
        description="Fish Audio API Key（Bearer Token）",
        json_schema_extra={
            "label": "Fish Audio API Key",
            "input_type": "password",
            "order": 1,
            "hint": "在 Fish Audio 控制台创建并复制 API 密钥",
        },
    )
    api_base_url: str = Field(
        default="https://api.fish.audio/v1/tts",
        description="Fish Audio TTS 接口地址（支持填写基础域名或完整 /v1/tts 路径）",
        json_schema_extra={"label": "接口地址", "order": 2},
    )
    model: Literal["s2.1-pro-free", "s2.1-pro", "s2-pro", "drama-3", "s1"] = Field(
        default="s2.1-pro-free",
        description="Fish Audio 合成模型（通过 HTTP 请求头 model 字段传递）",
        json_schema_extra={"label": "TTS 模型", "order": 3},
    )
    voice_mode: Literal["auto", "reference_id", "clone", "default"] = Field(
        default="auto",
        description=(
            "音色选择策略：auto（依次尝试云端 reference_id、本地 voices 克隆、模型默认）、"
            "reference_id（仅使用云端音色 ID）、clone（仅使用本地参考音频克隆）、default（模型默认音色）"
        ),
        json_schema_extra={"label": "音色模式", "order": 4},
    )
    reference_id: str = Field(
        default="",
        description="Fish Audio 云端音色 reference_id（留空使用模型默认；支持逗号分隔多音色如 id1,id2）",
        json_schema_extra={"label": "云端音色 ID (reference_id)", "order": 5},
    )
    voices_dir: str = Field(
        default="voices",
        description="本地音色参考目录（支持相对插件目录或绝对路径；可放置 .wav/.mp3/.flac 与同名 .txt 台词文本）",
        json_schema_extra={"label": "本地音色目录", "order": 6},
    )
    clone_voice: str = Field(
        default="",
        description="指定本地克隆音色名（不含扩展名；留空时优先选 default，否则取目录内首个音频）",
        json_schema_extra={"label": "指定本地音色名", "order": 7},
    )
    audio_format: Literal["mp3", "wav", "pcm", "opus"] = Field(
        default="mp3",
        description="向 Fish Audio 请求的输出音频格式（聊天平台发送语音推荐 mp3）",
        json_schema_extra={"label": "输出音频格式", "order": 8},
    )
    sample_rate: int = Field(
        default=0,
        ge=0,
        le=48000,
        description="输出采样率（0 表示沿用服务端默认：WAV/PCM 为 44100，Opus 为 48000）",
        json_schema_extra={"label": "采样率", "order": 9},
    )
    mp3_bitrate: Literal[64, 128, 192] = Field(
        default=128,
        description="MP3 比特率（kbps，仅 format=mp3 生效）",
        json_schema_extra={"label": "MP3 比特率", "order": 10},
    )
    opus_bitrate: Literal[-1000, 24000, 32000, 48000, 64000] = Field(
        default=-1000,
        description="Opus 比特率（-1000 表示自动选择，仅 format=opus 生效）",
        json_schema_extra={"label": "Opus 比特率", "order": 11},
    )


class GenerationSectionConfig(PluginConfigBase):
    """Fish Audio 合成与韵律控制参数。"""

    __ui_label__ = "合成与韵律参数"

    latency: Literal["balanced", "normal", "low"] = Field(
        default="balanced",
        description="延迟模式：normal 注重完整质量，balanced 均衡，low 优先降低延迟",
        json_schema_extra={"label": "延迟模式", "order": 1},
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="采样随机度（0.0-1.0）",
        json_schema_extra={"label": "采样温度 (temperature)", "order": 2},
    )
    top_p: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="核采样阈值（0.0-1.0）",
        json_schema_extra={"label": "Top-P", "order": 3},
    )
    repetition_penalty: float = Field(
        default=1.2,
        ge=0.5,
        le=2.0,
        description="重复惩罚系数",
        json_schema_extra={"label": "重复惩罚 (repetition_penalty)", "order": 4},
    )
    chunk_length: int = Field(
        default=300,
        ge=100,
        le=300,
        description="每次迭代的参考分块长度（100-300）",
        json_schema_extra={"label": "分块长度 (chunk_length)", "order": 5},
    )
    min_chunk_length: int = Field(
        default=50,
        ge=0,
        le=100,
        description="触发分块的最小字符阈值（0-100）",
        json_schema_extra={"label": "最小分块长度", "order": 6},
    )
    normalize: bool = Field(
        default=True,
        description="是否对中英文数字、日期、缩写等文本执行语音规范化",
        json_schema_extra={"label": "文本规范化 (normalize)", "order": 7},
    )
    language: str = Field(
        default="",
        description="BCP-47 语言提示代码（如 zh、en、ja；留空自动检测）",
        json_schema_extra={"label": "语言提示 (language)", "order": 8},
    )
    condition_on_previous_chunks: bool = Field(
        default=True,
        description="多段长文本是否继承上一分块的声学韵律",
        json_schema_extra={"label": "跨分块韵律连贯", "order": 9},
    )
    max_new_tokens: int = Field(
        default=1024,
        ge=64,
        le=4096,
        description="单次请求最多生成的语音 token 数",
        json_schema_extra={"label": "最大 Token 数", "order": 10},
    )
    early_stop_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="静音早停置信阈值（0.0-1.0）",
        json_schema_extra={"label": "早停阈值", "order": 11},
    )
    prosody_speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Fish Audio 服务端语速倍率（0.5-2.0，不改变音高）",
        json_schema_extra={"label": "云端语速 (prosody.speed)", "order": 12},
    )
    prosody_volume: float = Field(
        default=0.0,
        ge=-20.0,
        le=20.0,
        description="Fish Audio 服务端音量调节（dB）",
        json_schema_extra={"label": "云端音量 (prosody.volume)", "order": 13},
    )
    prosody_normalize_loudness: bool = Field(
        default=True,
        description="是否在服务端启用响度归一化",
        json_schema_extra={"label": "云端响度归一化", "order": 14},
    )


class RequestSectionConfig(PluginConfigBase):
    """网络请求超时、重试与并发控制。"""

    __ui_label__ = "超时与并发控制"

    request_timeout: float = Field(
        default=90.0,
        ge=5.0,
        le=600.0,
        description="单次 HTTP 请求总超时时间（秒）",
        json_schema_extra={"label": "总请求超时（秒）", "order": 1},
    )
    connect_timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="TCP 与 TLS 建连超时时间（秒）",
        json_schema_extra={"label": "建连超时（秒）", "order": 2},
    )
    stall_timeout: float = Field(
        default=30.0,
        ge=0.0,
        le=300.0,
        description="读取响应体的静默卡死超时时间（秒，0 表示不限制）",
        json_schema_extra={"label": "读包静默超时（秒）", "order": 3},
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="遇到限流或临时服务错误（429/5xx）时的最大重试次数",
        json_schema_extra={"label": "最大重试次数", "order": 4},
    )
    retry_backoff: float = Field(
        default=1.5,
        ge=0.5,
        le=10.0,
        description="重试退避基数（秒）",
        json_schema_extra={"label": "退避基数（秒）", "order": 5},
    )
    max_concurrency: int = Field(
        default=1,
        ge=1,
        le=8,
        description="同时允许进行 TTS 合成的最大并发数",
        json_schema_extra={"label": "最大并发数", "order": 6},
    )


class FallbackSectionConfig(PluginConfigBase):
    """失败退回纯文本与并发保护配置。"""

    __ui_label__ = "降级与文本保护"

    fallback_to_text: bool = Field(
        default=True,
        description="语音合成或发送失败时，自动改发纯文本消息",
        json_schema_extra={"label": "失败退回文字", "order": 1},
    )
    empty_text_placeholder: str = Field(
        default="（想说点什么，但声音没发出来）",
        description="剥离情感标签后正文为空时使用的保底文字",
        json_schema_extra={"label": "空台词保底文本", "order": 2},
    )
    strip_audio_tags: bool = Field(
        default=True,
        description="退回发送文字前，剥离 [laughing] / （轻笑）等副语言舞台指示",
        json_schema_extra={"label": "文字剥离情感标签", "order": 3},
    )
    skip_when_busy: bool = Field(
        default=True,
        description="当前并发槽位已满时，不阻塞排队，直接降级发文字",
        json_schema_extra={"label": "并发繁忙直接降级", "order": 4},
    )


class AudioSectionConfig(PluginConfigBase):
    """本地 ffmpeg 后处理参数（默认全关，零额外耗时）。"""

    __ui_label__ = "ffmpeg 本地调音"

    ffmpeg_path: str = Field(
        default="ffmpeg",
        description="ffmpeg 可执行文件路径或命令名",
        json_schema_extra={"label": "ffmpeg 路径", "order": 1},
    )
    pitch_semitones: float = Field(
        default=0.0,
        ge=-12.0,
        le=12.0,
        description="本地变调半音数（0.0 关闭；正数变细，负数变沉）",
        json_schema_extra={"label": "变调半音", "order": 2},
    )
    preserve_formant: bool = Field(
        default=True,
        description="变调时是否优先使用 rubberband 保持共振峰自然度",
        json_schema_extra={"label": "保持共振峰", "order": 3},
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="本地 ffmpeg 语速倍率（1.0 关闭；云端语速优先使用 prosody_speed）",
        json_schema_extra={"label": "本地语速", "order": 4},
    )
    loudness_normalize: bool = Field(
        default=False,
        description="是否开启本地 ffmpeg EBU R128 响度归一化",
        json_schema_extra={"label": "本地响度归一化", "order": 5},
    )
    loudness_target: float = Field(
        default=-16.0,
        ge=-30.0,
        le=-6.0,
        description="响度归一化目标 LUFS 值",
        json_schema_extra={"label": "目标响度 (LUFS)", "order": 6},
    )
    highpass_hz: int = Field(
        default=0,
        ge=0,
        le=500,
        description="高通滤波截止频率 Hz（0 关闭，常用 80 去低频闷音）",
        json_schema_extra={"label": "高通滤波 (Hz)", "order": 7},
    )
    volume_db: float = Field(
        default=0.0,
        ge=-24.0,
        le=24.0,
        description="本地额外音量增益（dB，0.0 关闭）",
        json_schema_extra={"label": "本地音量增益 (dB)", "order": 8},
    )
    fade_ms: int = Field(
        default=0,
        ge=0,
        le=2000,
        description="首尾淡入淡出毫秒数（0 关闭）",
        json_schema_extra={"label": "淡入淡出 (ms)", "order": 9},
    )
    trim_silence: bool = Field(
        default=False,
        description="是否裁切首尾多余静音",
        json_schema_extra={"label": "裁切首尾静音", "order": 10},
    )


class AutoVoiceSectionConfig(PluginConfigBase):
    """语音触发策略、Planner 自主决策、概率控制与分段发送配置。"""

    __ui_label__ = "语音触发与分段"

    mode: Literal["planner", "probability", "always", "off"] = Field(
        default="planner",
        description=(
            "语音发送触发模式：planner（由 Planner 根据对话语境自行决定是否发语音）、"
            "probability（按设定概率随机将符合条件的纯文本转为语音）、"
            "always（所有符合条件的出站纯文本全部转为语音）、"
            "off（关闭自动与 Planner 触发，仅保留 API 或手动调用）"
        ),
        json_schema_extra={
            "label": "语音触发模式",
            "order": 1,
            "hint": "推荐 planner：由 Planner 自主识别语境决定是否发语音；也可选纯随机概率、全量或关闭",
            "i18n": {
                "zh-CN": {
                    "choice:planner": "Planner 自主判断 (planner)",
                    "choice:probability": "纯随机概率 (probability)",
                    "choice:always": "全局全转语音 (always)",
                    "choice:off": "关闭自动触发 (off)",
                }
            },
        },
    )
    planner_temperature: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Planner 模式下控制模型主动选用语音的倾向温度（0.0 - 1.0）："
            "0.0 仅当用户明确要求语音时触发；0.5 均衡自然；1.0 极高倾向语音"
        ),
        json_schema_extra={
            "label": "Planner 温度滑块",
            "x-widget": "slider",
            "step": 0.05,
            "order": 2,
            "hint": "仅 mode=planner 生效；数值越高，Planner 越积极启用语音回复，同时放宽契合度阈值",
        },
    )
    random_probability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="纯随机概率模式（mode=probability）下出站纯文本自动转为语音的概率（0.0 - 1.0）",
        json_schema_extra={
            "label": "纯随机概率滑块",
            "x-widget": "slider",
            "step": 0.05,
            "order": 3,
            "hint": "仅 mode=probability 生效；0.0 不转换，0.5 约半数转换，1.0 全部转换",
        },
    )
    decision_ttl_seconds: float = Field(
        default=8.0,
        ge=0.0,
        le=60.0,
        description="纯随机概率模式下同一会话在多少秒内复用同一次随机决策（避免一条长回复拆段后语音与文字混发，0 表示关闭缓存）",
        json_schema_extra={"label": "随机决策缓存时长（秒）", "order": 4},
    )
    split_sentences: bool = Field(
        default=True,
        description="遇到多句或较长文本时按句拆分为多条语音依次分段发送",
        json_schema_extra={"label": "长文本分段发送", "order": 5},
    )
    max_segment_chars: int = Field(
        default=80,
        ge=15,
        le=300,
        description="分段发送时单条语音允许的最大字符数（超出则按标点继续拆段）",
        json_schema_extra={"label": "单段最大字数", "order": 6},
    )
    segment_delay_ms: int = Field(
        default=200,
        ge=0,
        le=3000,
        description="连续发送多段语音时每段之间的间隔毫秒数（0 表示无额外等待）",
        json_schema_extra={"label": "分段发送间隔 (ms)", "order": 7},
    )
    min_text_length: int = Field(
        default=1,
        ge=1,
        le=50,
        description="自动/扩展转为语音的最短文本字符数",
        json_schema_extra={"label": "最小转换字数", "order": 8},
    )
    max_text_length: int = Field(
        default=500,
        ge=20,
        le=4000,
        description="允许转为语音的单条原始消息最大总字数（超出则保持纯文本避免超时）",
        json_schema_extra={"label": "最大转换字数", "order": 9},
    )
    skip_urls_and_code: bool = Field(
        default=True,
        description="消息正文包含网址链接或代码块时保持纯文本不转语音",
        json_schema_extra={"label": "跳过链接与代码块", "order": 10},
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_enabled(cls, data: Any) -> Any:
        """兼容旧版 [auto_voice].enabled 布尔开关，平滑迁移至 mode。"""
        if not isinstance(data, Mapping):
            return data
        copied = dict(data)
        if "mode" not in copied and "enabled" in copied:
            copied["mode"] = "always" if bool(copied.get("enabled")) else "off"
        copied.pop("enabled", None)
        return copied


class StorageSectionConfig(PluginConfigBase):
    """输出语音文件存储路径与自动清理策略。"""

    __ui_label__ = "语音文件存储与清理"

    save_output_audio: bool = Field(
        default=False,
        description=(
            "是否将每次生成的语音文件保存到本地磁盘；默认 false（仅在内存通过 Base64 传输，"
            "不在硬盘留下语音缓存；开启本地 ffmpeg 调音时的临时目录也会在调音结束后立即自动删除）"
        ),
        json_schema_extra={"label": "保存输出语音文件", "order": 1},
    )
    output_dir: str = Field(
        default="output_audio",
        description="开启保存后输出语音文件的储存目录（支持相对插件目录或绝对路径）",
        json_schema_extra={"label": "语音储存目录", "order": 2},
    )
    auto_delete: bool = Field(
        default=True,
        description="开启本地保存后，按保留时长与数量上限自动清理历史插件输出音频（仅删除本插件生成的 fishaudio_* 文件）",
        json_schema_extra={"label": "自动清理输出文件", "order": 3},
    )
    retain_minutes: int = Field(
        default=60,
        ge=0,
        le=43200,
        description="输出语音文件保留时长（分钟）；0 表示发送完成后立即删除，>0 表示定期清理超时文件",
        json_schema_extra={"label": "保留时长（分钟）", "order": 4},
    )
    max_saved_files: int = Field(
        default=100,
        ge=0,
        le=5000,
        description="输出目录最多保留的语音文件数量（0 表示不限制；超出时从最旧文件开始自动删除）",
        json_schema_extra={"label": "最大保留文件数", "order": 5},
    )


class EmotionSectionConfig(PluginConfigBase):
    """Fish Audio 行内情绪与语气标签自动注入配置。"""

    __ui_label__ = "情绪与语气自动注入"

    enabled: bool = Field(
        default=True,
        description="是否在语音合成前自动分析台词并将情绪与语气标签（如 [开心的语气]、[平静]、[语气激动]、[哼歌]）注入到语句中",
        json_schema_extra={"label": "启用情绪参数自动注入", "order": 1},
    )
    mode: Literal["hybrid", "llm", "rule"] = Field(
        default="hybrid",
        description=(
            "情绪注入模式：hybrid（优先调用宿主大模型细腻标注，超时或失败时自动回退本地规则）、"
            "llm（仅使用宿主大模型智能标注，失败时保留原文）、"
            "rule（纯本地启发式规则零延迟注入，不调用大模型）"
        ),
        json_schema_extra={
            "label": "情绪注入模式",
            "order": 2,
            "i18n": {
                "zh-CN": {
                    "choice:hybrid": "智能混合：LLM 优先 + 规则兜底 (hybrid)",
                    "choice:llm": "仅宿主大模型注入 (llm)",
                    "choice:rule": "仅本地规则零延迟注入 (rule)",
                }
            },
        },
    )
    model_task: str = Field(
        default="",
        description="调用宿主 ctx.llm.generate 时指定的模型任务名或具体模型名（留空使用宿主默认模型任务；也可填 utils、replyer 等）",
        json_schema_extra={
            "label": "LLM 模型任务名 (model_task)",
            "order": 3,
            "placeholder": "留空使用默认，或填 utils",
        },
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="情绪标注大模型的采样温度（0.0 - 1.0，越低越稳定保守，越高情绪变化越丰富）",
        json_schema_extra={
            "label": "情绪标注温度",
            "x-widget": "slider",
            "step": 0.05,
            "order": 4,
        },
    )
    timeout_seconds: float = Field(
        default=8.0,
        ge=1.0,
        le=30.0,
        description="单次 LLM 情绪标注的最大等待秒数（超时立即无感回退，不阻塞语音合成）",
        json_schema_extra={"label": "LLM 标注超时（秒）", "order": 5},
    )
    tag_presets: str = Field(
        default="开心的语气, 平静, 语气激动, 温柔, 撒娇, 哼歌, 小声说话, 叹气, 轻笑, 疑惑, 委屈, 严肃, 俏皮, 惊讶",
        description="推荐使用的情绪、语气与副语言标签参考列表（逗号分隔，会作为优先词汇提供给标注器；Fish Audio S2 系列同时支持开放式自然语言标签）",
        json_schema_extra={"label": "常用情绪标签参考库", "order": 6},
    )
    custom_instruction: str = Field(
        default="",
        description="可选的额外配音导演指令（例如限定角色性格基调：整体偏俏皮活泼，句尾喜欢哼歌），留空按通用规则标注",
        json_schema_extra={"label": "自定义配音导演附加要求", "order": 7},
    )
    skip_if_already_tagged: bool = Field(
        default=True,
        description="若待合成台词中已包含 [标签] 或 (标签) 形式的情绪标记，则直接沿用不再重复注入",
        json_schema_extra={"label": "已带标签时跳过注入", "order": 8},
    )
    show_injected_text: bool = Field(
        default=True,
        description="是否在语音消息元信息、WebUI 观察面板及出站日志中显示带 [情绪标签] 的实际转换文本（语音失败降级发文字时仍会自动剥离标签）",
        json_schema_extra={"label": "展示带标签的转换文本", "order": 9},
    )
    send_injected_text_to_chat: bool = Field(
        default=False,
        description="是否在发送语音时同步在聊天窗口发一条带 [情绪标签] 的实际转换文本预览（便于边听语音边核对标签效果）",
        json_schema_extra={"label": "聊天窗口同步发送转换文本", "order": 10},
    )


class FishAudioPluginConfig(PluginConfigBase):
    """Fish Audio TTS 插件顶层配置模型。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    voice: VoiceSectionConfig = Field(default_factory=VoiceSectionConfig)
    auto_voice: AutoVoiceSectionConfig = Field(default_factory=AutoVoiceSectionConfig)
    emotion: EmotionSectionConfig = Field(default_factory=EmotionSectionConfig)
    storage: StorageSectionConfig = Field(default_factory=StorageSectionConfig)
    generation: GenerationSectionConfig = Field(default_factory=GenerationSectionConfig)
    request: RequestSectionConfig = Field(default_factory=RequestSectionConfig)
    fallback: FallbackSectionConfig = Field(default_factory=FallbackSectionConfig)
    audio: AudioSectionConfig = Field(default_factory=AudioSectionConfig)


@dataclass(frozen=True)
class LocalVoiceEntry:
    """本地音色文件索引项。"""

    name: str
    audio_path: Path
    transcript_path: Optional[Path] = None


def strip_audio_tags(text: str) -> str:
    """剥离台词中的 <style>、[laughing]、（轻笑）等副语言舞台指示标签。"""
    cleaned = _STYLE_BLOCK_RE.sub("", text or "")
    cleaned = _SQUARE_TAG_RE.sub("", cleaned)
    cleaned = _CN_PAREN_TAG_RE.sub("", cleaned)
    cleaned = _EN_PAREN_TAG_RE.sub("", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    lines = [line.strip() for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def apply_style_instruction(text: str, style_instruction: str, model: str) -> str:
    """根据 Fish Audio 模型语法将显式风格指令附加到台词前。

    - s1 模型支持圆括号情感提示，如 ``(whispering) 你好``；
    - s2 / s2.1 / drama-3 系列使用方括号标签，如 ``[whispering] 你好``；
    - 当未显式传入 ``style_instruction`` 时保持原台词不变。
    """
    raw = (text or "").strip()
    hint = (style_instruction or "").strip()
    if not raw or not hint:
        return raw

    normalized_hint = _STYLE_BLOCK_RE.sub("", hint).strip()
    normalized_hint = normalized_hint.strip("[]()（）").strip()
    if not normalized_hint:
        return raw

    if (model or "").strip().lower() == "s1":
        tag = f"({normalized_hint})"
    else:
        tag = f"[{normalized_hint}]"

    if raw.startswith(tag):
        return raw
    return f"{tag} {raw}"


_CODE_FENCE_RE = re.compile(r"^```[^\r\n]*\r?\n(.*?)\r?\n```$", re.DOTALL)
_INJECT_PREFIX_RE = re.compile(r"^(?:注入后(?:文本|结果)?|标注后(?:文本|结果)?|修改后(?:文本|结果)?|输出|结果)\s*[：:]\s*")
_RULE_SENT_BOUNDARY_RE = re.compile(
    r"(?:[。！？!?]+[”’\"'」』）)]*|(?<!\d)\.(?!\d)(?=\s|$)[”’\"'」』）)]*|[~～]+(?=\s|$)|\r?\n)"
)
_REPEAT_AAA_OR_AABB_RE = re.compile(r"([\u4e00-\u9fff])\1\1|([\u4e00-\u9fff])\2([\u4e00-\u9fff])\3")
_RULE_HUMMING_KEYWORDS = (
    "嘿咻嘿咻",
    "啦啦啦",
    "啦啦",
    "哼哼",
    "哒哒哒",
    "噜噜",
    "嘟嘟噜",
    "叮咚",
    "咻咻",
)
_RULE_HAPPY_PREFIX_RE = re.compile(
    r"^(?:嘿嘿|哈哈|嘻嘻|好耶|太棒|太好|哇塞|哇啊|哇|耶|快看|超开心|真开心)"
)
_RULE_MID_EXCITED_RE = re.compile(
    r"(?:对着你|冲着你|朝你|向你|让你|把你|给你|就要|非要|立刻|马上|直接|疯狂|使劲|拼命)[^\u4e00-\u9fff]{0,2}"
    r"(?=([\u4e00-\u9fff])\1\1|([\u4e00-\u9fff])\2([\u4e00-\u9fff])\3)"
)
_RULE_CONDITIONAL_HEAD_RE = re.compile(
    r"^(?:如果|只要|只有|因为|既然|虽然|其实|本来|要是|假如|万一|不然|否则)"
)
_RULE_STRONG_EXCITED_WORDS = (
    "天呐",
    "天哪",
    "救命",
    "好家伙",
    "气死",
    "烦死",
    "可恶",
    "绝了",
    "好厉害",
    "太强了",
    "冲鸭",
    "千万别",
    "绝对不行",
)


def format_emotion_tag(tag_body: str, model: str = "s2.1-pro-free") -> str:
    """按目标 Fish Audio 模型语法格式化单个情绪/语气标签。"""
    cleaned = _STYLE_BLOCK_RE.sub("", str(tag_body or "")).strip()
    cleaned = cleaned.strip("[]()（）").strip()
    if not cleaned:
        return ""
    if (model or "").strip().lower() == "s1":
        return f"({cleaned})"
    return f"[{cleaned}]"


def has_inline_emotion_tags(text: str) -> bool:
    """判断文本中是否已包含行内方括号情绪标签。"""
    return bool(_SQUARE_TAG_RE.search(text or ""))


def _visible_tts_length(text: str) -> int:
    """计算剥离行内方括号标签后的可朗读净文本长度。"""
    return len(_SQUARE_TAG_RE.sub("", text or ""))


def _extract_inline_tags(
    text: str,
    *,
    reference_plain: str = "",
    model: str = "s2.1-pro-free",
) -> tuple[str, dict[int, list[str]]]:
    """从候选文本中剥离行内标签，并返回 (去标签纯文本, {纯文本偏移: [规范化标签...]})。"""
    raw = text or ""
    matches: list[tuple[int, int, str]] = [
        (m.start(), m.end(), m.group(0)[1:-1]) for m in _SQUARE_TAG_RE.finditer(raw)
    ]
    if not matches:
        for pattern in (_EN_PAREN_TAG_RE, _CN_PAREN_TAG_RE):
            for m in pattern.finditer(raw):
                snippet = m.group(0)
                # 仅提取原台词中原本不存在的新增括号标签，防止误伤正文括号补充说明
                if not reference_plain or snippet not in reference_plain:
                    matches.append((m.start(), m.end(), snippet[1:-1]))
        matches.sort(key=lambda item: item[0])

    plain_chunks: list[str] = []
    tags_by_pos: dict[int, list[str]] = {}
    plain_len = 0
    cursor = 0
    for start, end, body in matches:
        if start < cursor:
            continue
        if start > cursor:
            segment = raw[cursor:start]
            plain_chunks.append(segment)
            plain_len += len(segment)
        formatted = format_emotion_tag(body, model=model)
        if formatted:
            tags_by_pos.setdefault(plain_len, []).append(formatted)
        cursor = end

    if cursor < len(raw):
        plain_chunks.append(raw[cursor:])

    return "".join(plain_chunks), tags_by_pos


def _clean_llm_emotion_candidate(candidate_text: str, original_text: str) -> str:
    """清洗 LLM 返回结果中可能掺杂的代码块围栏、提示前缀或多余外层引号。"""
    cleaned = (candidate_text or "").strip()
    fence_match = _CODE_FENCE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    prefix_match = _INJECT_PREFIX_RE.match(cleaned)
    if prefix_match and not _INJECT_PREFIX_RE.match(original_text.strip()):
        cleaned = cleaned[prefix_match.end() :].strip()

    for left_q, right_q in (('"', '"'), ("'", "'"), ("“", "”"), ("「", "」")):
        if (
            len(cleaned) >= 2
            and cleaned.startswith(left_q)
            and cleaned.endswith(right_q)
            and not (original_text.startswith(left_q) and original_text.endswith(right_q))
        ):
            cleaned = cleaned[1:-1].strip()
            break
    return cleaned


def reconcile_injected_text(
    original_text: str,
    candidate_text: str,
    model: str = "s2.1-pro-free",
) -> Optional[str]:
    """校对并锚定注入标签后的文本，确保原始台词一字不改；差异过大则返回 None。"""
    orig = original_text or ""
    if not orig.strip():
        return None

    cleaned_cand = _clean_llm_emotion_candidate(candidate_text, orig)
    if not cleaned_cand:
        return None

    cand_plain, tags_by_cand_pos = _extract_inline_tags(
        cleaned_cand,
        reference_plain=orig,
        model=model,
    )
    if not tags_by_cand_pos:
        return None

    cand_compact = re.sub(r"\s+", "", cand_plain)
    orig_compact = re.sub(r"\s+", "", orig)
    if not cand_compact:
        return None

    matcher = difflib.SequenceMatcher(None, cand_plain, orig)
    if cand_plain != orig:
        compact_ratio = difflib.SequenceMatcher(None, cand_compact, orig_compact).ratio()
        if max(matcher.ratio(), compact_ratio) < 0.75:
            return None

    cand_to_orig: list[int] = [len(orig)] * (len(cand_plain) + 1)
    for tag_op, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag_op == "equal":
            for k in range(i2 - i1):
                cand_to_orig[i1 + k] = j1 + k
        elif tag_op in ("replace", "delete"):
            for idx in range(i1, i2):
                cand_to_orig[idx] = j1
    cand_to_orig[len(cand_plain)] = len(orig)

    tags_by_orig: dict[int, list[str]] = {}
    for cand_pos, tag_list in sorted(tags_by_cand_pos.items()):
        orig_pos = cand_to_orig[min(cand_pos, len(cand_plain))]
        # 忽略句末没有任何后续字符的悬空标签
        if orig_pos >= len(orig):
            continue
        bucket = tags_by_orig.setdefault(orig_pos, [])
        for tag in tag_list:
            if not bucket or bucket[-1] != tag:
                bucket.append(tag)

    if not tags_by_orig:
        return None

    parts: list[str] = []
    last_emitted_tag = ""
    for idx, ch in enumerate(orig):
        for tag in tags_by_orig.get(idx, ()):
            if tag != last_emitted_tag:
                parts.append(tag)
                last_emitted_tag = tag
        parts.append(ch)
    return "".join(parts)


def _split_sentence_spans(text: str) -> list[str]:
    """按句末标点与换行切分文本并 100% 保留原始字符与空白。"""
    spans: list[str] = []
    last_end = 0
    for match in _RULE_SENT_BOUNDARY_RE.finditer(text):
        end = match.end()
        if end > last_end:
            spans.append(text[last_end:end])
        last_end = end
    if last_end < len(text):
        spans.append(text[last_end:])
    return spans


def _analyze_sentence_emotion(core: str, *, is_first: bool) -> tuple[str, Optional[tuple[int, str]]]:
    """基于语义与标点特征分析单个句子的开场情绪及可选句中激动突变位置。"""
    trimmed = core.rstrip("”’\"'」』）) \t")
    body = core.rstrip("。！？!?~～”’\"'」』）) \t")

    if any(kw in body for kw in _RULE_HUMMING_KEYWORDS):
        return "哼歌", None
    if trimmed.endswith(("~", "～")) and not any(q in core for q in ("？", "?")) and len(body) <= 24:
        if re.search(r"([\u4e00-\u9fff])\1|([\u4e00-\u9fff]{2})\2", body):
            return "哼歌", None

    if _RULE_HAPPY_PREFIX_RE.match(body) and not _RULE_CONDITIONAL_HEAD_RE.match(body):
        return "开心的语气", None
    if any(kw in body for kw in ("好开心", "超开心", "太开心", "真开心", "太棒啦", "太好啦")):
        if not any(neg in body for neg in ("不开心", "难过", "委屈")):
            return "开心的语气", None

    if any(kw in body for kw in ("噗嗤", "呵呵", "偷笑", "忍不住笑")):
        return "轻笑", None
    if re.match(r"^(?:唉|算了吧|好累|好无奈)", body) or "叹气" in body:
        return "叹气", None
    if any(kw in body for kw in ("呜呜", "呜哇", "好委屈", "欺负我", "好难过")):
        return "委屈", None
    if any(kw in body for kw in ("偷偷告诉", "小声", "悄悄", "嘘")):
        return "小声说话", None
    if any(kw in body for kw in ("晚安", "好梦", "早点睡", "抱抱", "摸摸头", "辛苦啦", "别难过")):
        return "温柔", None
    if trimmed.endswith(("？", "?")):
        return "疑惑", None

    is_exclam = trimmed.endswith(("！", "!"))
    if is_exclam:
        mid_match = _RULE_MID_EXCITED_RE.search(core)
        if mid_match is not None and mid_match.end() >= 4:
            return "平静", (mid_match.end(), "语气激动")

        if _RULE_CONDITIONAL_HEAD_RE.match(body):
            comma_idx = max(core.find("，"), core.find(","))
            if comma_idx >= 3:
                after_comma = core[comma_idx + 1 :]
                dup_match = _REPEAT_AAA_OR_AABB_RE.search(after_comma)
                if dup_match is not None:
                    return "平静", (comma_idx + 1 + dup_match.start(), "语气激动")

        is_long_conditional = bool(_RULE_CONDITIONAL_HEAD_RE.match(body)) and len(body) > 14
        if not is_long_conditional:
            if "！！" in core or "!!" in core:
                return "语气激动", None
            if len(body) <= 18 and _REPEAT_AAA_OR_AABB_RE.search(body):
                return "语气激动", None
            if any(kw in body for kw in _RULE_STRONG_EXCITED_WORDS):
                return "语气激动", None
            if len(body) <= 10 and any(kw in body for kw in ("太棒", "太好", "冲鸭", "加油", "可恶", "讨厌")):
                return "语气激动", None

    if trimmed.endswith("。") or _RULE_CONDITIONAL_HEAD_RE.match(body) or len(body) >= 12:
        return "平静", None

    if is_exclam:
        return "开心的语气", None
    return ("平静" if is_first else ""), None


def inject_emotion_tags_by_rules(text: str, model: str = "s2.1-pro-free") -> str:
    """使用零延迟本地启发式规则为台词注入行内 Fish Audio 情绪/语气标签。"""
    raw = text or ""
    if not raw.strip() or not _SPEAKABLE_CHAR_RE.search(raw):
        return raw

    spans = _split_sentence_spans(raw)
    if not spans:
        return raw

    out_chunks: list[str] = []
    current_emotion = ""
    seen_speakable = False

    for span in spans:
        ws_len = len(span) - len(span.lstrip())
        leading_ws = span[:ws_len]
        core = span[ws_len:]
        if not core or not _SPEAKABLE_CHAR_RE.search(core):
            out_chunks.append(span)
            continue

        head_emotion, mid_shift = _analyze_sentence_emotion(core, is_first=not seen_speakable)
        seen_speakable = True

        head_prefix = ""
        if head_emotion and head_emotion != current_emotion:
            head_prefix = format_emotion_tag(head_emotion, model=model)
            current_emotion = head_emotion

        if mid_shift is not None:
            shift_idx, mid_emotion = mid_shift
            mid_tag = format_emotion_tag(mid_emotion, model=model)
            if mid_tag and 0 < shift_idx < len(core):
                annotated_core = f"{head_prefix}{core[:shift_idx]}{mid_tag}{core[shift_idx:]}"
                current_emotion = mid_emotion
            else:
                annotated_core = f"{head_prefix}{core}"
        else:
            annotated_core = f"{head_prefix}{core}"

        out_chunks.append(f"{leading_ws}{annotated_core}")

    return "".join(out_chunks)


def map_annotated_to_segments(
    segments: Sequence[str],
    annotated: str,
    model: str = "s2.1-pro-free",
) -> list[str]:
    """将完整注入后的带标签文本精确拆回各个分段，并跨段自动延续活跃情绪标签。"""
    seg_list = list(segments)
    if not seg_list:
        return []

    joined = "\n".join(seg_list)
    plain, tags_by_pos = _extract_inline_tags(
        annotated,
        reference_plain=joined,
        model=model,
    )
    if plain != joined:
        return [inject_emotion_tags_by_rules(seg, model=model) for seg in seg_list]

    out_segments: list[str] = []
    cursor = 0
    active_tag = ""
    carried_tags: list[str] = []

    for seg_idx, seg in enumerate(seg_list):
        seg_start = cursor
        seg_end = seg_start + len(seg)
        seg_parts: list[str] = []
        prev_active = active_tag

        start_tags = list(carried_tags) + list(tags_by_pos.get(seg_start, ()))
        carried_tags = []

        if start_tags:
            for tag in start_tags:
                if not seg_parts or seg_parts[-1] != tag:
                    seg_parts.append(tag)
                    active_tag = tag
        elif prev_active and _SPEAKABLE_CHAR_RE.search(seg):
            seg_parts.append(prev_active)

        for offset_in_seg, ch in enumerate(seg):
            abs_pos = seg_start + offset_in_seg
            if offset_in_seg > 0:
                for tag in tags_by_pos.get(abs_pos, ()):
                    if tag != active_tag:
                        seg_parts.append(tag)
                        active_tag = tag
            seg_parts.append(ch)

        if seg_idx + 1 < len(seg_list):
            carried_tags = list(tags_by_pos.get(seg_end, ()))
            cursor = seg_end + 1
        else:
            cursor = seg_end

        out_segments.append("".join(seg_parts))

    return out_segments


def _read_transcript(path: Optional[Path], logger: Optional[logging.Logger] = None) -> str:
    """安全读取伴随参考台词 .txt 文件。"""
    if path is None or not path.is_file():
        return ""
    try:
        if path.stat().st_size > _MAX_REFERENCE_TEXT_BYTES:
            if logger:
                logger.warning("参考文本 %s 超过大小上限，已跳过读取", path.name)
            return ""
        raw = path.read_bytes()
    except OSError as exc:
        if logger:
            logger.warning("读取参考文本 %s 失败: %s", path.name, exc)
        return ""

    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore").strip()


def _extract_stream_id(record: Any) -> str:
    """从消息或聊天记录对象中提取聊天流 ID。"""
    if record is None:
        return ""
    if isinstance(record, dict):
        for key in ("stream_id", "chat_id", "session_id"):
            val = str(record.get(key) or "").strip()
            if val:
                return val
        return ""
    for key in ("stream_id", "chat_id", "session_id"):
        val = str(getattr(record, key, "") or "").strip()
        if val:
            return val
    return ""


def _resolve_plugin_path(raw_dir: str, default_rel: str) -> Path:
    """将插件配置目录解析为规范绝对路径，相对路径基于插件目录。"""
    cleaned = str(raw_dir or "").strip() or default_rel
    candidate = Path(cleaned)
    if not candidate.is_absolute():
        candidate = PLUGIN_DIR / candidate
    return candidate.resolve()


def _tokenize_with_inline_tags(text: str) -> list[tuple[bool, str]]:
    """将文本拆分为 (是否为行内标签, 片段) 序列，避免硬切分时截断 [标签]。"""
    tokens: list[tuple[bool, str]] = []
    pos = 0
    for match in _SQUARE_TAG_RE.finditer(text):
        if match.start() > pos:
            for ch in text[pos : match.start()]:
                tokens.append((False, ch))
        tokens.append((True, match.group(0)))
        pos = match.end()
    if pos < len(text):
        for ch in text[pos:]:
            tokens.append((False, ch))
    return tokens


def _hard_split_preserving_tags(piece: str, limit: int) -> list[str]:
    """按净可见字符上限切分超长分片，同时保护 [标签] 完整不被劈开。"""
    if not _SQUARE_TAG_RE.search(piece):
        return [piece[idx : idx + limit].strip() for idx in range(0, len(piece), limit) if piece[idx : idx + limit].strip()]

    chunks: list[str] = []
    buf: list[str] = []
    vis_len = 0
    for is_tag, token in _tokenize_with_inline_tags(piece):
        if is_tag:
            if vis_len >= limit and buf:
                chunk = "".join(buf).strip()
                if chunk:
                    chunks.append(chunk)
                buf = []
                vis_len = 0
            buf.append(token)
            continue

        if vis_len >= limit and buf:
            chunk = "".join(buf).strip()
            if chunk:
                chunks.append(chunk)
            buf = []
            vis_len = 0
        buf.append(token)
        vis_len += 1

    if buf:
        chunk = "".join(buf).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _pack_fragments(pieces: list[str], limit: int) -> list[str]:
    """将细粒片段按最大长度合并打包，超长单片再做硬切分。"""
    packed: list[str] = []
    current = ""
    for raw_piece in pieces:
        piece = raw_piece.strip()
        if not piece:
            continue
        if _visible_tts_length(piece) > limit:
            if current:
                packed.append(current)
                current = ""
            packed.extend(_hard_split_preserving_tags(piece, limit))
            continue
        if not current:
            current = piece
        elif _visible_tts_length(current) + 1 + _visible_tts_length(piece) <= limit:
            separator = " " if current[-1].isascii() and piece[0].isascii() else ""
            current = f"{current}{separator}{piece}"
        else:
            packed.append(current)
            current = piece
    if current:
        packed.append(current)
    return packed


def split_tts_segments(
    text: str,
    *,
    max_chars: int = 80,
    split_sentences: bool = True,
) -> list[str]:
    """将待合成文本按段落、句子和长度拆分为适合分段语音发送的片段列表。"""
    cleaned = (text or "").strip()
    if not cleaned or not _SPEAKABLE_CHAR_RE.search(cleaned):
        return []

    if not split_sentences:
        return [cleaned]

    limit = max(15, int(max_chars or 80))
    raw_sentences: list[str] = []
    for paragraph in cleaned.splitlines():
        para = paragraph.strip()
        if not para:
            continue
        for sent in _SENTENCE_SPLIT_RE.split(para):
            s = sent.strip()
            if s:
                raw_sentences.append(s)

    segments: list[str] = []
    for sentence in raw_sentences:
        if _visible_tts_length(sentence) <= limit:
            if _SPEAKABLE_CHAR_RE.search(sentence):
                segments.append(sentence)
            continue
        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(sentence) if c.strip()]
        for packed in _pack_fragments(clauses or [sentence], limit):
            if _SPEAKABLE_CHAR_RE.search(packed):
                segments.append(packed)

    return segments


def _collect_at_names_from_segments(segments: Any) -> list[str]:
    """从消息段列表提取 @ 段对应的昵称列表，避免正文朗读冗余 @ 称呼。"""
    names: list[str] = []
    if not isinstance(segments, list):
        return names
    for item in segments:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").strip().lower() != "at":
            continue
        data = item.get("data")
        if not isinstance(data, dict):
            continue
        for key in ("target_user_cardname", "target_user_nickname"):
            value = str(data.get(key) or "").strip()
            if value and value not in names:
                names.append(value)
    return names


def _extract_plain_text_from_segments(segments: Any) -> Optional[str]:
    """提取可转为语音的纯文本；遇到语音/图片/文件/视频等非文本段返回 None。"""
    if not isinstance(segments, list) or not segments:
        return None

    parts: list[str] = []
    for item in segments:
        if not isinstance(item, dict):
            return None
        seg_type = str(item.get("type") or "").strip().lower()
        if seg_type in _NON_BODY_SEGMENTS:
            continue
        if seg_type == "dict":
            data_obj = item.get("data")
            nested_type = (
                str(data_obj.get("type") or "").strip().lower()
                if isinstance(data_obj, dict)
                else ""
            )
            if nested_type in _MEDIA_DICT_TYPES or not nested_type:
                return None
            continue
        if seg_type != "text":
            return None
        raw_data = item.get("data")
        if isinstance(raw_data, dict):
            parts.append(str(raw_data.get("text") or ""))
        else:
            parts.append(str(raw_data or ""))

    text = "".join(parts).strip()
    if not text:
        return None

    for name in _collect_at_names_from_segments(segments):
        mention = f"@{name}"
        if mention in text:
            text = text.replace(mention, " ")
    normalized = _MULTI_SPACE_RE.sub(" ", text).strip()
    return normalized or None


def _extract_outbound_plain_text(message: dict[str, Any]) -> Optional[str]:
    """提取出站消息中可转为语音的纯文本；遇到非文本媒体段返回 None。"""
    return _extract_plain_text_from_segments(message.get("raw_message"))


def _has_voice_segment(raw_message: Any) -> bool:
    """判断消息段列表是否已包含语音或语音链接段。"""
    if not isinstance(raw_message, list):
        return False
    return any(
        isinstance(item, dict)
        and str(item.get("type") or "").strip().lower() in {"voice", "voiceurl"}
        for item in raw_message
    )


def _resolve_temperature_tier(temperature: float) -> tuple[str, str]:
    """按温度值匹配对应的 Planner 语音倾向档位描述与引导文案。"""
    clamped = max(0.0, min(1.0, float(temperature)))
    for max_temp, label, guidance in _PLANNER_TEMPERATURE_TIERS:
        if clamped <= max_temp + 1e-9:
            return label, guidance
    _, label, guidance = _PLANNER_TEMPERATURE_TIERS[-1]
    return label, guidance


def _should_allow_planner_voice(
    temperature: float,
    *,
    explicit_request: bool,
    suitability: float,
) -> bool:
    """结合 Planner 温度滑块与语境契合度决定是否放行语音回复。"""
    clamped_temp = max(0.0, min(1.0, float(temperature)))
    clamped_score = max(0.0, min(1.0, float(suitability)))
    if explicit_request:
        return True
    if clamped_temp <= 0.0:
        return False
    if clamped_temp >= 1.0:
        return True
    min_suitability = 1.0 - clamped_temp
    return clamped_score + 1e-9 >= min_suitability


def _find_voice_extension_key(tool_definitions: Any) -> str:
    """从 reply 工具声明的 plugin_options 中动态定位本插件回复扩展键名。"""
    if not isinstance(tool_definitions, list):
        return _DEFAULT_VOICE_EXTENSION_FULL_NAME
    for tool_def in tool_definitions:
        if not isinstance(tool_def, dict):
            continue
        fn = tool_def.get("function")
        if not isinstance(fn, dict) or fn.get("name") != "reply":
            continue
        params = fn.get("parameters")
        if not isinstance(params, dict):
            continue
        props = params.get("properties")
        if not isinstance(props, dict):
            continue
        plugin_opts = props.get("plugin_options")
        if not isinstance(plugin_opts, dict):
            continue
        ext_props = plugin_opts.get("properties")
        if not isinstance(ext_props, dict):
            continue
        for key in ext_props:
            key_str = str(key)
            if key_str == _DEFAULT_VOICE_EXTENSION_FULL_NAME or (
                key_str.endswith(f".{_VOICE_REPLY_EXTENSION_NAME}") and "fishaudio" in key_str.lower()
            ):
                return key_str
    return _DEFAULT_VOICE_EXTENSION_FULL_NAME


def _build_planner_voice_instruction(temperature: float, extension_full_name: str) -> str:
    """构造注入给 MaiSaka Planner 的动态语音策略指引。"""
    clamped = max(0.0, min(1.0, float(temperature)))
    tier_label, tier_guidance = _resolve_temperature_tier(clamped)
    return (
        "【Fish Audio 语音回复策略】\n"
        f"当前语音倾向温度：{clamped:.2f}（{tier_label}）。{tier_guidance}\n"
        f"需要发送语音时：优先在 reply 工具的 plugin_options[\"{extension_full_name}\"] 中启用"
        "（被动发言模式则直接在发言指导中说明使用语音并注明风格），也可仅在需要独立短语音时直接调用 send_voice_reply；"
        "切勿对同一条回复同时输出文字与语音。\n"
        "参考边界：适合语音（情绪安慰、撒娇关心、道别问候、接梗吐槽、对方刚发来语音或明确要求语音）；"
        "建议保持文字（超过两三句的长内容、分点列举、原理/步骤解释、含网址链接或代码块、以及上一条刚连续用过语音的普通应答）。"
    )


def _filter_voice_tools_for_non_planner(tool_definitions: Any) -> tuple[list[Any], bool]:
    """在非 planner 模式下从候选工具声明中移除语音工具及 reply 的语音扩展选项。"""
    if not isinstance(tool_definitions, list):
        return [], False

    changed = False
    filtered: list[Any] = []
    for raw_tool in tool_definitions:
        if not isinstance(raw_tool, dict):
            filtered.append(raw_tool)
            continue
        fn = raw_tool.get("function")
        fn_name = fn.get("name") if isinstance(fn, dict) else None
        if fn_name == "send_voice_reply":
            changed = True
            continue

        if fn_name == "reply" and isinstance(fn, dict):
            params = fn.get("parameters")
            props = params.get("properties") if isinstance(params, dict) else None
            plugin_opts = props.get("plugin_options") if isinstance(props, dict) else None
            ext_props = plugin_opts.get("properties") if isinstance(plugin_opts, dict) else None
            if isinstance(ext_props, dict):
                to_remove = [
                    str(k)
                    for k in ext_props
                    if str(k) == _DEFAULT_VOICE_EXTENSION_FULL_NAME
                    or (str(k).endswith(f".{_VOICE_REPLY_EXTENSION_NAME}") and "fishaudio" in str(k).lower())
                ]
                if to_remove:
                    tool_copy = copy.deepcopy(raw_tool)
                    copy_props = tool_copy["function"]["parameters"]["properties"]
                    copy_ext_props = copy_props["plugin_options"]["properties"]
                    for k in to_remove:
                        copy_ext_props.pop(k, None)
                    if not copy_ext_props:
                        copy_props.pop("plugin_options", None)
                    filtered.append(tool_copy)
                    changed = True
                    continue

        filtered.append(raw_tool)
    return filtered, changed


def _strip_planner_instruction_items(items: list[Any]) -> tuple[list[Any], bool]:
    """移除此前注入的 Fish Audio Planner 语音策略提示 Item。"""
    cleaned: list[Any] = []
    removed = False
    for item in items:
        if isinstance(item, dict):
            meta = item.get("meta")
            item_id = str(meta.get("item_id") or "") if isinstance(meta, dict) else ""
            if item_id.startswith(_PLANNER_VOICE_ITEM_PREFIX):
                removed = True
                continue
        cleaned.append(item)
    return cleaned, removed


def _upsert_planner_instruction_item(
    items: list[Any],
    session_id: str,
    instruction: str,
) -> tuple[list[Any], bool]:
    """向 Planner Context Items 的最后一条用户消息前插入或更新语音策略提示。"""
    cleaned, _ = _strip_planner_instruction_items(items)
    target_id = f"{_PLANNER_VOICE_ITEM_PREFIX}{session_id or 'default'}"
    new_item = {
        "item_type": "SystemMessageItem",
        "meta": {
            "item_id": target_id,
            "logical_turn_id": None,
            "timestamp": datetime.now().isoformat(),
        },
        "parts": [{"type": "text", "text": instruction}],
    }
    insert_at = len(cleaned)
    for index in range(len(cleaned) - 1, -1, -1):
        entry = cleaned[index]
        if isinstance(entry, dict) and entry.get("item_type") == "UserMessageItem":
            insert_at = index
            break
    updated = [*cleaned[:insert_at], new_item, *cleaned[insert_at:]]
    return updated, updated != items


def _build_voice_hook_result(
    kwargs: dict[str, Any],
    message: dict[str, Any],
    audio_base64: str,
    spoken_text: str,
) -> dict[str, Any]:
    """构造替换出站正文为原生 voice 消息段的 HookResult。"""
    updated_message = dict(message)
    updated_message["raw_message"] = [
        {
            "type": "voice",
            "data": spoken_text,
            "binary_data_base64": audio_base64,
        }
    ]
    updated_message["processed_plain_text"] = spoken_text

    updated_kwargs = dict(kwargs)
    updated_kwargs["message"] = updated_message
    if kwargs.get("set_reply"):
        updated_kwargs["set_reply"] = False
        updated_kwargs["reply_message_id"] = ""
    return {"action": "continue", "modified_kwargs": updated_kwargs}


def _build_text_override_hook_result(
    kwargs: dict[str, Any],
    message: dict[str, Any],
    remaining_text: str,
) -> dict[str, Any]:
    """当多段语音局部失败时，用剩余未合成文本替换当前待发文本。"""
    updated_message = dict(message)
    updated_message["raw_message"] = [{"type": "text", "data": remaining_text}]
    updated_message["processed_plain_text"] = remaining_text
    updated_kwargs = dict(kwargs)
    updated_kwargs["message"] = updated_message
    return {"action": "continue", "modified_kwargs": updated_kwargs}


def _prune_output_dir_sync(
    output_dir: Path,
    *,
    retain_minutes: int,
    max_saved_files: int,
) -> int:
    """安全清理输出目录中本插件生成的历史语音文件（仅匹配 fishaudio_*）。"""
    if not output_dir.is_dir():
        return 0

    resolved_root = output_dir.resolve()
    now_ts = time.time()
    max_age_sec = max(0, int(retain_minutes)) * 60
    kept: list[tuple[float, Path]] = []
    removed = 0

    try:
        entries = list(resolved_root.iterdir())
    except OSError:
        return 0

    for item in entries:
        try:
            if item.is_symlink() or not item.is_file():
                continue
            resolved_item = item.resolve()
            if not resolved_item.is_relative_to(resolved_root):
                continue
            if not _OUTPUT_FILE_NAME_RE.match(resolved_item.name):
                continue
            stat = resolved_item.stat()
        except OSError:
            continue

        if max_age_sec == 0 or (now_ts - stat.st_mtime) > max_age_sec:
            try:
                resolved_item.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
        else:
            kept.append((stat.st_mtime, resolved_item))

    limit = int(max_saved_files)
    if limit > 0 and len(kept) > limit:
        kept.sort(key=lambda pair: pair[0])
        for _, victim in kept[: len(kept) - limit]:
            try:
                victim.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass

    return removed


def _save_and_prune_output_audio_sync(
    output_dir: Path,
    audio_bytes: bytes,
    audio_format: str,
    *,
    auto_delete: bool,
    retain_minutes: int,
    max_saved_files: int,
) -> Optional[Path]:
    """将输出音频保存至配置目录，并在启用自动清理时回收过期/超量文件。"""
    ext = (audio_format or "mp3").strip().lower()
    if ext not in {"mp3", "wav", "opus", "pcm"}:
        ext = "mp3"

    resolved_root = output_dir.resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = (resolved_root / f"fishaudio_{stamp}_{uuid.uuid4().hex[:8]}.{ext}").resolve()
    if not target.is_relative_to(resolved_root):
        return None
    target.write_bytes(audio_bytes)

    if auto_delete and retain_minutes > 0:
        _prune_output_dir_sync(
            resolved_root,
            retain_minutes=retain_minutes,
            max_saved_files=max_saved_files,
        )
    return target


def _delete_saved_output_sync(path: Optional[Path], output_dir: Path) -> None:
    """发送完毕后立即删除单个输出文件（用于 retain_minutes == 0）。"""
    if path is None:
        return
    try:
        resolved_root = output_dir.resolve()
        resolved_item = path.resolve()
        if (
            not path.is_symlink()
            and resolved_item.is_file()
            and resolved_item.is_relative_to(resolved_root)
            and _OUTPUT_FILE_NAME_RE.match(resolved_item.name)
        ):
            resolved_item.unlink(missing_ok=True)
    except OSError:
        pass


class FishAudioVoicePlugin(MaiBotPlugin):
    """Fish Audio TTS 与即时音色克隆插件实现。"""

    config_model = FishAudioPluginConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tts_service: Optional[FishAudioTTSService] = None
        self._audio_processor: Optional[AudioProcessor] = None
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(1)
        self._max_concurrency: int = 1
        self._fallback_bypass: dict[tuple[str, str], float] = {}
        self._random_decisions: dict[str, tuple[bool, float]] = {}

    def normalize_plugin_config(self, config_data: Mapping[str, Any] | None) -> tuple[dict[str, Any], bool]:
        """在 SDK 默认值补齐前迁移旧版 [auto_voice].enabled 至 mode。"""
        if not isinstance(config_data, Mapping):
            return super().normalize_plugin_config(config_data)

        prepared = copy.deepcopy(dict(config_data))
        raw_auto = prepared.get("auto_voice")
        migrated = False
        if isinstance(raw_auto, Mapping) and "mode" not in raw_auto and "enabled" in raw_auto:
            auto_copy = dict(raw_auto)
            auto_copy["mode"] = "always" if bool(auto_copy.pop("enabled")) else "off"
            prepared["auto_voice"] = auto_copy
            migrated = True

        normalized, changed = super().normalize_plugin_config(prepared)
        return normalized, changed or migrated

    def _safe_ctx(self) -> Any:
        try:
            return self.ctx
        except RuntimeError:
            return None

    @property
    def _log(self) -> logging.Logger:
        ctx_logger = getattr(self._safe_ctx(), "logger", None)
        if ctx_logger is not None:
            return ctx_logger
        return logging.getLogger("fishaudio.plugin")

    def _is_enabled(self) -> bool:
        try:
            cfg = self.config
        except RuntimeError:
            return True
        plugin_sec = getattr(cfg, "plugin", None)
        return bool(getattr(plugin_sec, "enabled", True))

    def _auto_voice_mode(self) -> str:
        av = getattr(self.config, "auto_voice", None)
        if av is None:
            return "off"
        mode = str(getattr(av, "mode", "") or "").strip().lower()
        if mode in _AUTO_VOICE_MODES:
            return mode
        if hasattr(av, "enabled"):
            return "always" if bool(getattr(av, "enabled")) else "off"
        return "planner"

    def _voices_dir(self) -> Path:
        raw_dir = str(getattr(getattr(self.config, "voice", None), "voices_dir", "") or "voices")
        return _resolve_plugin_path(raw_dir, "voices")

    def _output_dir(self) -> Path:
        raw_dir = str(getattr(getattr(self.config, "storage", None), "output_dir", "") or "output_audio")
        return _resolve_plugin_path(raw_dir, "output_audio")

    def _work_dir(self) -> Optional[Path]:
        paths = getattr(self._safe_ctx(), "paths", None)
        runtime_dir = getattr(paths, "runtime_dir", None) if paths is not None else None
        if runtime_dir:
            return Path(runtime_dir)
        data_dir = getattr(paths, "data_dir", None) if paths is not None else None
        if data_dir:
            return Path(data_dir)
        return None

    def _rebuild_runtime(self) -> Optional[FishAudioTTSService]:
        """按当前配置重建服务客户端与音频处理器，返回旧客户端供异步关闭。"""
        cfg: FishAudioPluginConfig = self.config
        old_service = self._tts_service
        self._random_decisions.clear()

        self._max_concurrency = max(1, int(cfg.request.max_concurrency or 1))
        self._semaphore = asyncio.Semaphore(self._max_concurrency)

        stall = float(cfg.request.stall_timeout or 0.0)
        self._tts_service = FishAudioTTSService(
            api_key=cfg.voice.fish_api_key,
            api_base_url=cfg.voice.api_base_url,
            logger=self._log,
            request_timeout=float(cfg.request.request_timeout),
            connect_timeout=float(cfg.request.connect_timeout),
            max_retries=int(cfg.request.max_retries),
            retry_backoff=float(cfg.request.retry_backoff),
            max_concurrency=self._max_concurrency,
            stall_timeout=stall if stall > 0 else None,
        )
        self._audio_processor = build_audio_processor(
            cfg,
            logger=self._log,
            work_dir=self._work_dir(),
        )
        return old_service

    async def _run_storage_prune(self) -> None:
        storage = getattr(self.config, "storage", None)
        if storage is None or not storage.auto_delete:
            return
        try:
            removed = await asyncio.to_thread(
                _prune_output_dir_sync,
                self._output_dir(),
                retain_minutes=int(storage.retain_minutes),
                max_saved_files=int(storage.max_saved_files),
            )
            if removed > 0:
                self._log.info("已清理 %d 个过期/超量 Fish Audio 输出语音文件", removed)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("清理输出语音目录失败: %s", exc)

    async def on_load(self) -> None:
        self._rebuild_runtime()
        await self._run_storage_prune()
        voices = self._discover_local_voices()
        cfg = self.config
        has_key = bool(str(cfg.voice.fish_api_key or "").strip())
        self._log.info(
            "Fish Audio TTS 插件已加载 (enabled=%s, auto_mode=%s, planner_temp=%.2f, random_prob=%.2f, emotion=%s/%s, model=%s, voice_mode=%s, has_key=%s, local_voices=%d)",
            self._is_enabled(),
            self._auto_voice_mode(),
            float(cfg.auto_voice.planner_temperature),
            float(cfg.auto_voice.random_probability),
            bool(cfg.emotion.enabled),
            str(cfg.emotion.mode),
            cfg.voice.model,
            cfg.voice.voice_mode,
            has_key,
            len(voices),
        )

    async def on_unload(self) -> None:
        await self._run_storage_prune()
        service = self._tts_service
        self._tts_service = None
        if service is not None:
            await service.close()
        self._log.info("Fish Audio TTS 插件已卸载")

    async def on_config_update(
        self,
        scope: str = CONFIG_RELOAD_SCOPE_SELF,
        config_data: Any = None,
        version: str = "",
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        del config_data
        if scope != CONFIG_RELOAD_SCOPE_SELF:
            return
        old_service = self._rebuild_runtime()
        if old_service is not None:
            await old_service.close()
        await self._run_storage_prune()
        self._log.info(
            "Fish Audio TTS 配置已热重载 (version=%s, enabled=%s, auto_mode=%s, planner_temp=%.2f, random_prob=%.2f, emotion=%s/%s, model=%s, voice_mode=%s)",
            str(version or self.config.plugin.config_version),
            self._is_enabled(),
            self._auto_voice_mode(),
            float(self.config.auto_voice.planner_temperature),
            float(self.config.auto_voice.random_probability),
            bool(self.config.emotion.enabled),
            str(self.config.emotion.mode),
            self.config.voice.model,
            self.config.voice.voice_mode,
        )

    def _discover_local_voices(self) -> dict[str, LocalVoiceEntry]:
        """扫描本地 voices 目录，建立安全的文件名到音频/台词路径映射。"""
        vdir = self._voices_dir()
        if not vdir.is_dir():
            return {}

        entries: dict[str, LocalVoiceEntry] = {}
        try:
            candidates = sorted(vdir.iterdir(), key=lambda p: p.name.lower())
        except OSError as exc:
            self._log.warning("扫描本地音色目录失败 (%s): %s", vdir, exc)
            return {}

        for item in candidates:
            if not item.is_file() or item.suffix.lower() not in _AUDIO_EXTENSIONS:
                continue
            try:
                resolved = item.resolve()
            except OSError:
                continue
            if not resolved.is_relative_to(vdir):
                continue

            stem = item.stem.strip()
            if not stem or stem in entries:
                continue

            txt_candidate = (item.with_suffix(".txt")).resolve()
            transcript_path: Optional[Path] = None
            if txt_candidate.is_file() and txt_candidate.is_relative_to(vdir):
                transcript_path = txt_candidate

            entries[stem] = LocalVoiceEntry(
                name=stem,
                audio_path=resolved,
                transcript_path=transcript_path,
            )
        return entries

    def _load_local_reference(self, entry: LocalVoiceEntry) -> Optional[ReferenceSample]:
        """按索引项安全读取本地参考音频及伴随文本。"""
        try:
            size = entry.audio_path.stat().st_size
        except OSError as exc:
            self._log.warning("读取本地音色文件信息失败 (%s): %s", entry.audio_path.name, exc)
            return None

        if size <= 0:
            self._log.warning("本地音色文件为空 (%s)", entry.audio_path.name)
            return None
        if size > _MAX_REFERENCE_AUDIO_BYTES:
            self._log.warning("本地音色文件超过 25MB 上限，已跳过 (%s)", entry.audio_path.name)
            return None

        try:
            audio_bytes = entry.audio_path.read_bytes()
        except OSError as exc:
            self._log.warning("读取本地音色文件失败 (%s): %s", entry.audio_path.name, exc)
            return None

        transcript = _read_transcript(entry.transcript_path, logger=self._log)
        return ReferenceSample(audio=audio_bytes, text=transcript)

    def _pick_local_entry(
        self,
        local_voices: dict[str, LocalVoiceEntry],
        preferred: str = "",
    ) -> Optional[LocalVoiceEntry]:
        if not local_voices:
            return None

        target = (preferred or "").strip()
        if target:
            if target in local_voices:
                return local_voices[target]
            lowered_map = {k.lower(): v for k, v in local_voices.items()}
            if target.lower() in lowered_map:
                return lowered_map[target.lower()]
            return None

        if "default" in local_voices:
            return local_voices["default"]
        return next(iter(local_voices.values()))

    def _resolve_voice_target(
        self,
        voice_arg: str = "",
    ) -> tuple[str, str, Optional[list[ReferenceSample]]]:
        """解析本次合成应采用的音色策略。

        Returns:
            (描述标签, reference_id 字符串, 本地 ReferenceSample 列表或 None)
        """
        cfg = self.config.voice
        local_voices = self._discover_local_voices()
        explicit = (voice_arg or "").strip()

        if explicit:
            local_match = self._pick_local_entry(local_voices, explicit)
            if local_match is not None:
                sample = self._load_local_reference(local_match)
                if sample is not None:
                    return (f"clone:{local_match.name}", "", [sample])
            # 支持运行时临时传入云端 reference_id（含逗号分隔多 ID）
            return (f"reference_id:{explicit}", explicit, None)

        mode = str(cfg.voice_mode or "auto").strip().lower()
        configured_ref = str(cfg.reference_id or "").strip()
        configured_clone = str(cfg.clone_voice or "").strip()

        if mode == "default":
            return ("default", "", None)

        if mode == "reference_id":
            if configured_ref:
                return (f"reference_id:{configured_ref}", configured_ref, None)
            self._log.warning("voice_mode=reference_id 但未配置 reference_id，将回退为模型默认音色")
            return ("default", "", None)

        if mode == "clone":
            entry = self._pick_local_entry(local_voices, configured_clone)
            if entry is not None:
                sample = self._load_local_reference(entry)
                if sample is not None:
                    return (f"clone:{entry.name}", "", [sample])
            self._log.warning("voice_mode=clone 但未找到可用本地音色，将回退为模型默认音色")
            return ("default", "", None)

        # mode == "auto": reference_id -> 本地音色克隆 -> 模型默认
        if configured_ref:
            return (f"reference_id:{configured_ref}", configured_ref, None)

        entry = self._pick_local_entry(local_voices, configured_clone)
        if entry is not None:
            sample = self._load_local_reference(entry)
            if sample is not None:
                return (f"clone:{entry.name}", "", [sample])

        return ("default", "", None)

    def _build_synthesis_options(
        self,
        *,
        audio_format: str = "",
        speed: Optional[float] = None,
    ) -> FishSynthesisOptions:
        cfg = self.config
        fmt = (audio_format or cfg.voice.audio_format or "mp3").strip().lower()
        if fmt not in FishAudioTTSService.SUPPORTED_FORMATS:
            fmt = cfg.voice.audio_format or "mp3"

        eff_speed = cfg.generation.prosody_speed if speed is None else float(speed)
        eff_speed = max(0.5, min(2.0, eff_speed))

        return FishSynthesisOptions(
            model=cfg.voice.model,
            audio_format=fmt,
            sample_rate=int(cfg.voice.sample_rate or 0),
            mp3_bitrate=int(cfg.voice.mp3_bitrate or 128),
            opus_bitrate=int(cfg.voice.opus_bitrate if cfg.voice.opus_bitrate is not None else -1000),
            latency=cfg.generation.latency,
            temperature=float(cfg.generation.temperature),
            top_p=float(cfg.generation.top_p),
            repetition_penalty=float(cfg.generation.repetition_penalty),
            chunk_length=int(cfg.generation.chunk_length),
            min_chunk_length=int(cfg.generation.min_chunk_length),
            normalize=bool(cfg.generation.normalize),
            language=str(cfg.generation.language or "").strip(),
            condition_on_previous_chunks=bool(cfg.generation.condition_on_previous_chunks),
            max_new_tokens=int(cfg.generation.max_new_tokens),
            early_stop_threshold=float(cfg.generation.early_stop_threshold),
            prosody_speed=eff_speed,
            prosody_volume=float(cfg.generation.prosody_volume),
            prosody_normalize_loudness=bool(cfg.generation.prosody_normalize_loudness),
        )

    async def _post_process_audio(
        self,
        audio_base64: str,
        *,
        audio_format: str,
        sample_rate: int,
        wrap_pcm: bool,
    ) -> tuple[str, bytes, str]:
        """执行可选的 ffmpeg 后处理，并将裸 PCM 按需封装为 WAV。"""
        raw_bytes = base64.b64decode(audio_base64)
        fmt = (audio_format or "mp3").lower()
        effective_rate = int(sample_rate) if sample_rate and int(sample_rate) > 0 else DEFAULT_PCM_SAMPLE_RATE

        processed = raw_bytes
        if self._audio_processor is not None:
            processed = await self._audio_processor.process(
                raw_bytes,
                audio_format=fmt,
                sample_rate=effective_rate,
            )

        out_format = fmt
        if fmt in RAW_PCM_FORMATS and wrap_pcm:
            processed = wrap_raw_pcm_to_wav(processed, sample_rate=effective_rate)
            out_format = "wav"

        out_b64 = base64.b64encode(processed).decode("ascii")
        return out_b64, processed, out_format

    async def _maybe_save_audio(self, audio_bytes: bytes, out_format: str) -> Optional[Path]:
        """按 [storage] 配置决定是否将输出语音落盘并清理历史文件。"""
        storage = getattr(self.config, "storage", None)
        if storage is None or not storage.save_output_audio:
            return None
        try:
            return await asyncio.to_thread(
                _save_and_prune_output_audio_sync,
                self._output_dir(),
                audio_bytes,
                out_format,
                auto_delete=bool(storage.auto_delete),
                retain_minutes=int(storage.retain_minutes),
                max_saved_files=int(storage.max_saved_files),
            )
        except Exception as exc:  # noqa: BLE001
            self._log.warning("保存输出语音文件失败（不影响发送）: %s", exc)
            return None

    async def _maybe_delete_immediate(self, saved_path: Optional[Path]) -> None:
        """若配置 retain_minutes == 0 且启用自动删除，则在发送后立即清理该文件。"""
        storage = getattr(self.config, "storage", None)
        if saved_path is None or storage is None:
            return
        if storage.auto_delete and int(storage.retain_minutes) == 0:
            try:
                await asyncio.to_thread(_delete_saved_output_sync, saved_path, self._output_dir())
            except Exception as exc:  # noqa: BLE001
                self._log.warning("立即清理输出语音文件失败: %s", exc)

    def _build_emotion_prompt(self, text: str, *, model: str, style_hint: str = "") -> str:
        """构造用于 Fish Audio 行内情绪与语气标注的 LLM 提示词。"""
        emotion_cfg = self.config.emotion
        is_s1 = (model or "").strip().lower() == "s1"
        tag_syntax = "圆括号 (标签)" if is_s1 else "方括号 [标签]"
        presets = str(emotion_cfg.tag_presets or "").strip() or (
            "开心的语气, 平静, 语气激动, 温柔, 撒娇, 哼歌, 小声说话, 叹气, 轻笑, 疑惑"
        )
        example_in = (
            "嘿嘿，快看这里！如果你现在还没关注臭企鹅，那你的肚子肯定会变得圆滚滚的！"
            "圆圆滚滚！只有每天准时收看我的视频，才能变成最可爱的企鹅。"
            "要是错过了，我就要对着你滚滚滚，滚滚开开！嘿咻嘿咻~"
        )
        if is_s1:
            example_out = (
                "(开心的语气)嘿嘿，快看这里！(平静)如果你现在还没关注臭企鹅，那你的肚子肯定会变得圆滚滚的！"
                "(语气激动)圆圆滚滚！(平静)只有每天准时收看我的视频，才能变成最可爱的企鹅。"
                "要是错过了，我就要对着你(语气激动)滚滚滚，滚滚开开！(哼歌)嘿咻嘿咻~"
            )
        else:
            example_out = (
                "[开心的语气]嘿嘿，快看这里！[平静]如果你现在还没关注臭企鹅，那你的肚子肯定会变得圆滚滚的！"
                "[语气激动]圆圆滚滚！[平静]只有每天准时收看我的视频，才能变成最可爱的企鹅。"
                "要是错过了，我就要对着你[语气激动]滚滚滚，滚滚开开！[哼歌]嘿咻嘿咻~"
            )

        extra_lines: list[str] = []
        hint = (style_hint or "").strip()
        if hint:
            extra_lines.append(f"- 本次整体风格基调参考：{hint}")
        custom_instr = str(emotion_cfg.custom_instruction or "").strip()
        if custom_instr:
            extra_lines.append(f"- 用户自定义配音导演要求：{custom_instr}")
        extra_block = ("\n" + "\n".join(extra_lines)) if extra_lines else ""

        return (
            "你是专业配音导演，负责为即将送入 Fish Audio TTS 的台词注入行内情绪与语气控制标签。\n"
            "请严格遵守以下规则：\n"
            f"1. 仅使用 {tag_syntax} 格式插入情绪、语气或副语言标签，标签紧贴在对应句首或句中情绪转折词前方。\n"
            "2. 严禁修改、增删、替换或翻译原始台词中的任何汉字、英文、数字、标点或空白字符。\n"
            "3. 相邻句子情绪一致时不要重复打标；仅在开场或情绪发生起伏转折时插入新标签。\n"
            f"4. 常用标签参考（也可结合语境自然描述）：{presets}。{extra_block}\n"
            "5. 只输出注入标签后的完整文本，不要输出任何解释、引号或 Markdown 代码块。\n\n"
            f"示例输入：{example_in}\n"
            f"示例输出：{example_out}\n\n"
            f"待标注台词：{text}\n"
            "注入后输出："
        )

    async def _inject_emotion_tags_via_llm(
        self,
        text: str,
        *,
        model: str,
        style_hint: str = "",
    ) -> Optional[str]:
        """调用宿主 LLM 能力为台词执行行内情绪注入，并进行原文防篡改校验。"""
        ctx = self._safe_ctx()
        llm_cap = getattr(ctx, "llm", None) if ctx is not None else None
        generate_fn = getattr(llm_cap, "generate", None) if llm_cap is not None else None
        if not callable(generate_fn):
            return None

        emotion_cfg = self.config.emotion
        prompt = self._build_emotion_prompt(text, model=model, style_hint=style_hint)
        gen_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "temperature": max(0.0, min(1.0, float(emotion_cfg.temperature))),
            "max_tokens": max(256, min(2048, len(text) * 2 + 128)),
        }
        model_task = str(emotion_cfg.model_task or "").strip()
        if model_task:
            gen_kwargs["task_name"] = model_task

        timeout_sec = max(1.0, float(emotion_cfg.timeout_seconds or 8.0))
        try:
            result = await asyncio.wait_for(generate_fn(**gen_kwargs), timeout=timeout_sec)
        except asyncio.TimeoutError:
            self._log.warning("LLM 情绪参数注入超时（%.1fs），将按策略回退", timeout_sec)
            return None
        except Exception as exc:  # noqa: BLE001
            self._log.warning("LLM 情绪参数注入调用异常，将按策略回退: %s", exc)
            return None

        if not isinstance(result, dict) or not result.get("success"):
            err_msg = str(result.get("error") or "") if isinstance(result, dict) else ""
            if err_msg:
                self._log.debug("LLM 情绪参数注入未成功（%s），将按策略回退", err_msg)
            return None

        response_text = str(result.get("response") or "").strip()
        if not response_text:
            return None

        reconciled = reconcile_injected_text(text, response_text, model=model)
        if reconciled is None:
            self._log.warning("LLM 情绪注入结果未通过原文一致性校验，已回退")
            return None
        return reconciled

    async def inject_emotion_tags(
        self,
        text: str,
        *,
        model: str = "",
        style_hint: str = "",
        force_emotion: bool = False,
    ) -> str:
        """按配置（或在 force_emotion=True 时强制）将 Fish Audio 行内情绪标签注入台词。"""
        raw = (text or "").strip()
        if not raw or not _SPEAKABLE_CHAR_RE.search(raw):
            return text

        emotion_cfg = getattr(self.config, "emotion", None)
        if not force_emotion and (emotion_cfg is None or not emotion_cfg.enabled):
            return text

        skip_tagged = bool(getattr(emotion_cfg, "skip_if_already_tagged", True)) if emotion_cfg is not None else True
        if skip_tagged and has_inline_emotion_tags(raw):
            return text

        target_model = (model or self.config.voice.model or "s2.1-pro-free").strip()
        mode = str(getattr(emotion_cfg, "mode", "hybrid") or "hybrid").strip().lower()

        if mode in {"llm", "hybrid"}:
            via_llm = await self._inject_emotion_tags_via_llm(
                raw,
                model=target_model,
                style_hint=style_hint,
            )
            if via_llm:
                self._log.info("情绪参数注入完成 [LLM]: %s", via_llm)
                return via_llm

        if mode in {"rule", "hybrid"}:
            via_rule = inject_emotion_tags_by_rules(raw, model=target_model)
            self._log.info("情绪参数注入完成 [本地规则, mode=%s]: %s", mode, via_rule)
            return via_rule

        return text

    def _select_voice_display_text(self, item: dict[str, Any], *, force_emotion: bool = False) -> str:
        """根据配置决定写入语音消息段 data 的文本（展示带情绪标签的转换文本或纯净原文）。"""
        emotion_cfg = getattr(self.config, "emotion", None)
        if emotion_cfg is not None and (emotion_cfg.enabled or force_emotion) and emotion_cfg.show_injected_text:
            injected = str(item.get("injected_text") or "").strip()
            if injected:
                return injected
        return str(item.get("text") or "")

    def _should_send_injected_preview_to_chat(self, *, force_emotion: bool = False) -> bool:
        emotion_cfg = getattr(self.config, "emotion", None)
        return bool(
            emotion_cfg is not None
            and (emotion_cfg.enabled or force_emotion)
            and emotion_cfg.send_injected_text_to_chat
        )

    @staticmethod
    def _collect_injected_preview_text(items: Sequence[dict[str, Any]]) -> str:
        parts = [str(item.get("injected_text") or item.get("text") or "").strip() for item in items]
        return "\n".join(part for part in parts if part).strip()

    async def _maybe_send_injected_preview_to_chat(
        self,
        items: Sequence[dict[str, Any]],
        stream_id: str,
        *,
        force_emotion: bool = False,
    ) -> None:
        """当开启聊天窗口同步预览时，发出带 [情绪标签] 的实际转换文本且屏蔽自动语音环回。"""
        if not stream_id or not items or not self._should_send_injected_preview_to_chat(force_emotion=force_emotion):
            return
        preview_text = self._collect_injected_preview_text(items)
        if not preview_text:
            return
        self._register_fallback_bypass(stream_id, preview_text)
        try:
            await self.ctx.send.text(preview_text, stream_id)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("发送情绪注入文本预览失败 (stream=%s): %s", stream_id, exc)

    async def _prepare_emotion_segments(
        self,
        segments: Sequence[str],
        *,
        model: str = "",
        style_hint: str = "",
        force_emotion: bool = False,
    ) -> list[str]:
        """为多段待合成文本执行全局情绪标注并精确映射回各分段。"""
        seg_list = [str(s) for s in segments]
        if not seg_list:
            return []

        emotion_cfg = getattr(self.config, "emotion", None)
        if not force_emotion and (emotion_cfg is None or not emotion_cfg.enabled):
            return seg_list

        target_model = (model or self.config.voice.model or "s2.1-pro-free").strip()
        if len(seg_list) == 1:
            return [
                await self.inject_emotion_tags(
                    seg_list[0],
                    model=target_model,
                    style_hint=style_hint,
                    force_emotion=force_emotion,
                )
            ]

        skip_tagged = bool(getattr(emotion_cfg, "skip_if_already_tagged", True)) if emotion_cfg is not None else True
        if skip_tagged and any(has_inline_emotion_tags(s) for s in seg_list):
            return seg_list

        joined = "\n".join(seg_list)
        annotated = await self.inject_emotion_tags(
            joined,
            model=target_model,
            style_hint=style_hint,
            force_emotion=force_emotion,
        )
        if annotated == joined:
            return seg_list
        return map_annotated_to_segments(seg_list, annotated, model=target_model)

    async def _synthesize_pipeline(
        self,
        text: str,
        *,
        voice: str = "",
        style: str = "",
        audio_format: str = "",
        speed: Optional[float] = None,
        wrap_pcm: bool = True,
        defer_immediate_cleanup: bool = False,
        injected_text: Optional[str] = None,
        force_emotion: bool = False,
    ) -> dict[str, Any]:
        """共享合成主链路：服务调用、音频后处理、可选存储与结构化返回。"""
        if not self._is_enabled():
            return {"success": False, "error": "Fish Audio 插件当前已禁用", "busy": False}

        service = self._tts_service
        if service is None:
            self._rebuild_runtime()
            service = self._tts_service
        if service is None:
            return {"success": False, "error": "Fish Audio 服务未初始化", "busy": False}

        raw_text = (text or "").strip()
        if not raw_text:
            return {"success": False, "error": "合成文本为空", "busy": False}

        options = self._build_synthesis_options(audio_format=audio_format, speed=speed)
        if isinstance(injected_text, str) and injected_text.strip():
            prepared_text = injected_text.strip()
        else:
            prepared_text = await self.inject_emotion_tags(
                raw_text,
                model=options.model,
                style_hint=style,
                force_emotion=force_emotion,
            )

        is_s1 = (options.model or "").strip().lower() == "s1"
        has_injected_head_tag = (
            prepared_text != raw_text
            and prepared_text.startswith("(" if is_s1 else "[")
        )
        if style and not has_injected_head_tag:
            styled_text = apply_style_instruction(prepared_text, style, options.model)
        else:
            styled_text = prepared_text

        voice_label, ref_id, ref_samples = self._resolve_voice_target(voice)

        if self.config.fallback.skip_when_busy and self._semaphore.locked():
            return {
                "success": False,
                "error": "当前 TTS 并发队列繁忙，已触发快速跳过",
                "busy": True,
                "voice": voice_label,
            }

        async with self._semaphore:
            res = await service.synthesize(
                styled_text,
                options=options,
                reference_id=ref_id,
                references=ref_samples,
            )

        if not res.get("success"):
            return {
                "success": False,
                "error": str(res.get("error") or "未知合成错误"),
                "busy": False,
                "voice": voice_label,
            }

        out_b64, out_bytes, out_fmt = await self._post_process_audio(
            str(res["audio_base64"]),
            audio_format=options.audio_format,
            sample_rate=options.sample_rate,
            wrap_pcm=wrap_pcm,
        )
        saved_path = await self._maybe_save_audio(out_bytes, out_fmt)
        if saved_path is not None and not defer_immediate_cleanup:
            await self._maybe_delete_immediate(saved_path)
            storage = getattr(self.config, "storage", None)
            if storage is not None and storage.auto_delete and int(storage.retain_minutes) == 0:
                saved_path = None

        payload: dict[str, Any] = {
            "success": True,
            "audio_base64": out_b64,
            "audio_bytes_len": len(out_bytes),
            "format": out_fmt,
            "model": options.model,
            "voice": voice_label,
            "text": raw_text,
            "injected_text": styled_text,
        }
        if saved_path is not None:
            payload["audio_path"] = str(saved_path)
        return payload

    def _register_fallback_bypass(self, stream_id: str, text: str) -> None:
        now = time.monotonic()
        self._fallback_bypass = {k: exp for k, exp in self._fallback_bypass.items() if exp > now}
        self._fallback_bypass[(stream_id, text.strip())] = now + _FALLBACK_BYPASS_TTL_SECONDS

    def _consume_fallback_bypass(self, stream_id: str, text: str) -> bool:
        now = time.monotonic()
        self._fallback_bypass = {k: exp for k, exp in self._fallback_bypass.items() if exp > now}
        key = (stream_id, text.strip())
        if key in self._fallback_bypass:
            self._fallback_bypass.pop(key, None)
            return True
        return False

    async def _find_stream_id(self, kwargs: dict[str, Any]) -> str:
        """安全解析当前会话对应的 stream_id，杜绝跨群串话。"""
        for key in ("stream_id", "chat_id", "session_id"):
            direct = str(kwargs.get(key) or "").strip()
            if direct:
                return direct

        message_id = str(kwargs.get("message_id") or kwargs.get("reply_message_id") or "").strip()
        if message_id and hasattr(self.ctx, "message"):
            try:
                msg = await self.ctx.message.get_by_id(message_id, include_binary_data=False)
                from_msg = _extract_stream_id(msg)
                if from_msg:
                    return from_msg
            except Exception as exc:  # noqa: BLE001
                self._log.warning("按 message_id=%s 查询会话失败: %s", message_id, exc)

        platform = str(kwargs.get("platform") or "").strip()
        if hasattr(self.ctx, "chat"):
            try:
                streams = await self.ctx.chat.get_all_streams(platform=platform)
                if isinstance(streams, Sequence) and len(streams) == 1:
                    only_id = _extract_stream_id(streams[0])
                    if only_id:
                        return only_id
            except Exception as exc:  # noqa: BLE001
                self._log.warning("查询可用聊天流失败: %s", exc)

        return ""

    def _format_fallback_text(self, raw_text: str) -> str:
        """按降级配置清洗台词副语言标签并填充保底文本。"""
        fb = self.config.fallback
        cleaned = strip_audio_tags(raw_text) if fb.strip_audio_tags else (raw_text or "").strip()
        if cleaned:
            return cleaned
        return str(fb.empty_text_placeholder or "").strip() or "（想说点什么，但声音没发出来）"

    def _strip_tags_in_reply_messages(self, messages: list[dict[str, Any]]) -> Optional[list[dict[str, Any]]]:
        """当语音扩展降级保留文字时，按配置剥离仅用于 TTS 的情感舞台标签；无变化返回 None。"""
        if not self.config.fallback.strip_audio_tags:
            return None
        changed = False
        cleaned_messages: list[dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                cleaned_messages.append(msg)
                continue
            segments = msg.get("segments")
            if not isinstance(segments, list):
                cleaned_messages.append(msg)
                continue
            new_segments: list[dict[str, Any]] = []
            for seg in segments:
                if isinstance(seg, dict) and str(seg.get("type") or "").strip().lower() == "text":
                    orig = str(seg.get("data") or "")
                    stripped = self._format_fallback_text(orig)
                    if stripped != orig:
                        changed = True
                    new_segments.append({"type": "text", "data": stripped})
                else:
                    new_segments.append(seg)
            cleaned_messages.append({"segments": new_segments, "quote_previous": bool(msg.get("quote_previous", False))})
        return cleaned_messages if changed else None

    def _explain_ineligible_auto_voice_text(self, spoken_text: str) -> Optional[str]:
        """校验文本长度与内容是否符合自动/扩展转语音条件；不符合时返回原因说明。"""
        av = getattr(self.config, "auto_voice", None)
        if av is None or not spoken_text:
            return "空文本或缺少 auto_voice 配置"
        if len(spoken_text) < int(av.min_text_length):
            return f"长度 {len(spoken_text)} 小于 min_text_length={int(av.min_text_length)}"
        if len(spoken_text) > int(av.max_text_length):
            return f"长度 {len(spoken_text)} 超过 max_text_length={int(av.max_text_length)}"
        if av.skip_urls_and_code:
            for pat in _UNSUITABLE_TEXT_PATTERNS:
                if pat in spoken_text:
                    return f"包含跳过模式 {pat!r}"
        return None

    def _is_eligible_auto_voice_text(self, spoken_text: str) -> bool:
        """校验文本长度与内容是否符合自动/扩展转语音条件。"""
        return self._explain_ineligible_auto_voice_text(spoken_text) is None

    def _should_convert_by_probability(
        self,
        stream_id: str,
        probability: float,
        ttl_seconds: float,
    ) -> tuple[bool, bool]:
        """按随机概率与短时 TTL 缓存决定出站纯文本是否转为语音，返回 (是否转换, 是否命中缓存)。"""
        prob = max(0.0, min(1.0, float(probability)))
        now = time.monotonic()
        self._random_decisions = {sid: item for sid, item in self._random_decisions.items() if item[1] > now}

        if ttl_seconds > 0 and stream_id in self._random_decisions:
            decision, _ = self._random_decisions[stream_id]
            return decision, True

        if prob <= 0.0:
            decision = False
        elif prob >= 1.0:
            decision = True
        else:
            decision = random.random() < prob

        if ttl_seconds > 0:
            self._random_decisions[stream_id] = (decision, now + float(ttl_seconds))
        return decision, False

    async def _fallback_to_text(
        self,
        raw_text: str,
        stream_id: str,
        *,
        reason: str,
    ) -> dict[str, Any]:
        """合成或发送语音失败时降级发送纯文本。"""
        fb = self.config.fallback
        if not fb.fallback_to_text:
            return {
                "success": False,
                "fallback_sent": False,
                "error": reason,
            }

        if not stream_id:
            self._log.warning("无法解析目标 stream_id，无法执行文字降级（原因: %s）", reason)
            return {
                "success": False,
                "fallback_sent": False,
                "error": f"{reason}；且未能确定目标聊天流 stream_id",
            }

        text_to_send = self._format_fallback_text(raw_text)

        self._register_fallback_bypass(stream_id, text_to_send)
        try:
            send_res = await self.ctx.send.text(text_to_send, stream_id)
            ok = bool(getattr(send_res, "success", True))
            if not ok:
                err = str(getattr(send_res, "error", "") or "send.text 返回失败")
                self._log.error("文字降级发送失败 (stream=%s): %s", stream_id, err)
                return {"success": False, "fallback_sent": False, "error": f"{reason}; 降级发送失败: {err}"}
            self._log.info("已触发文字降级发送 (stream=%s, reason=%s)", stream_id, reason)
            return {
                "success": False,
                "fallback_sent": True,
                "sent_text": text_to_send,
                "error": reason,
            }
        except Exception as exc:  # noqa: BLE001
            self._log.error("文字降级发送异常 (stream=%s): %s", stream_id, exc)
            return {"success": False, "fallback_sent": False, "error": f"{reason}; 降级异常: {exc}"}

    async def _send_voice(self, audio_base64: str, stream_id: str) -> tuple[bool, str]:
        """调用 MaiBot send.custom 发送语音分片。"""
        try:
            send_res = await self.ctx.send.custom("voice", audio_base64, stream_id)
            if bool(getattr(send_res, "success", True)):
                return True, ""
            return False, str(getattr(send_res, "error", "") or "send.custom 返回失败")
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    async def _cleanup_deferred_files(self, items: list[dict[str, Any]]) -> None:
        """清理延迟立即删除列表中的输出语音文件。"""
        for item in items:
            saved_str = item.get("audio_path")
            if saved_str:
                await self._maybe_delete_immediate(Path(str(saved_str)))

    async def _synthesize_segments(
        self,
        segments: list[str],
        *,
        voice: str = "",
        style: str = "",
        total_budget_sec: Optional[float] = None,
        force_emotion: bool = False,
    ) -> tuple[list[dict[str, Any]], Optional[int], str]:
        """按顺序合成多个文本分片，受可选总预算约束，返回 (已成功列表, 首个失败索引, 失败原因)。"""
        synthesized: list[dict[str, Any]] = []
        deadline = time.monotonic() + total_budget_sec if total_budget_sec else None
        target_model = str(getattr(getattr(self.config, "voice", None), "model", "") or "s2.1-pro-free")

        prepared_segments = list(segments)
        try:
            prep_coro = self._prepare_emotion_segments(
                segments,
                model=target_model,
                style_hint=style,
                force_emotion=force_emotion,
            )
            if deadline is not None:
                prep_budget = max(0.1, deadline - time.monotonic())
                prepared_segments = await asyncio.wait_for(prep_coro, timeout=prep_budget)
            else:
                prepared_segments = await prep_coro
        except Exception:  # noqa: BLE001
            prepared_segments = list(segments)

        for idx, seg_text in enumerate(segments):
            injected_seg = prepared_segments[idx] if idx < len(prepared_segments) else seg_text
            coro = self._synthesize_pipeline(
                seg_text,
                voice=voice,
                style=style,
                wrap_pcm=True,
                defer_immediate_cleanup=True,
                injected_text=injected_seg,
                force_emotion=force_emotion,
            )
            try:
                if deadline is not None:
                    remaining_sec = deadline - time.monotonic()
                    if remaining_sec <= 0.5:
                        coro.close()
                        return synthesized, idx, f"默认语音总耗时接近钩子预算上限 ({total_budget_sec:.1f}s)"
                    res = await asyncio.wait_for(coro, timeout=remaining_sec)
                else:
                    res = await coro
            except asyncio.TimeoutError:
                return synthesized, idx, f"默认语音分段合成总耗时超出预算 ({total_budget_sec:.1f}s)"
            except Exception as exc:  # noqa: BLE001
                return synthesized, idx, f"分段语音合成异常: {exc}"

            if not res.get("success"):
                return synthesized, idx, str(res.get("error") or "语音合成失败")
            synthesized.append(res)
        return synthesized, None, ""

    async def _send_segment_sequence(
        self,
        items: list[dict[str, Any]],
        stream_id: str,
        *,
        delay_ms: int,
    ) -> tuple[int, str]:
        """按顺序发送多条已合成语音分片，返回成功条数与末次错误（若有）。"""
        sent_count = 0
        delay_sec = max(0, int(delay_ms)) / 1000.0
        for idx, item in enumerate(items):
            try:
                if idx > 0 and delay_sec > 0:
                    await asyncio.sleep(delay_sec)
                ok, send_err = await self._send_voice(str(item["audio_base64"]), stream_id)
            except Exception as exc:  # noqa: BLE001
                ok, send_err = False, str(exc)
            saved_str = item.get("audio_path")
            if saved_str:
                await self._maybe_delete_immediate(Path(str(saved_str)))
            if not ok:
                await self._cleanup_deferred_files(items[idx + 1 :])
                return sent_count, send_err
            sent_count += 1
        return sent_count, ""

    @HookHandler(
        "maisaka.planner.before_request",
        name="fish_planner_voice_policy",
        description="按当前语音触发模式与 Planner 温度滑块动态调整语音扩展可见性与倾向指引",
        mode=HookMode.BLOCKING,
        order=HookOrder.NORMAL,
        timeout_ms=2500,
        error_policy=ErrorPolicy.SKIP,
    )
    async def planner_before_request_hook(self, **kwargs: Any) -> Optional[dict[str, Any]]:
        """在 Planner 请求前注入语音决策策略，或在非 planner 模式下隐藏语音扩展。"""
        items = kwargs.get("items")
        tool_definitions = kwargs.get("tool_definitions")
        if not isinstance(items, list):
            return None

        active_planner = self._is_enabled() and self._auto_voice_mode() == "planner"
        updated_kwargs = dict(kwargs)
        mutated = False

        if not active_planner:
            cleaned_items, items_changed = _strip_planner_instruction_items(items)
            if items_changed:
                updated_kwargs["items"] = cleaned_items
                mutated = True
            if isinstance(tool_definitions, list):
                filtered_tools, tools_changed = _filter_voice_tools_for_non_planner(tool_definitions)
                if tools_changed:
                    updated_kwargs["tool_definitions"] = filtered_tools
                    mutated = True
            return {"action": "continue", "modified_kwargs": updated_kwargs} if mutated else None

        session_id = str(kwargs.get("session_id") or "").strip()
        av = self.config.auto_voice
        ext_full_name = _find_voice_extension_key(tool_definitions)
        instruction = _build_planner_voice_instruction(float(av.planner_temperature), ext_full_name)
        updated_items, items_changed = _upsert_planner_instruction_item(items, session_id, instruction)
        if items_changed:
            updated_kwargs["items"] = updated_items
            mutated = True
        return {"action": "continue", "modified_kwargs": updated_kwargs} if mutated else None

    @ReplyExtension(
        _VOICE_REPLY_EXTENSION_NAME,
        description=(
            "将本次回复通过 Fish Audio 合成为语音消息发送（仅适合口语互动或明确要求语音时启用，长文/代码/链接请保持文字）。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "use_voice": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否将该条回复转为语音发送；选定本扩展时通常为 true。",
                },
                "explicit_request": {
                    "type": "boolean",
                    "default": False,
                    "description": "是否为用户明确要求发语音（如“发条语音/念给我听”）或对方刚发来语音需要语音回应。",
                },
                "suitability": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.7,
                    "description": "自评当前语境使用语音的契合度（0.0-1.0）；情绪互动/撒娇安慰填 0.8-1.0，普通闲聊填 0.4-0.7。",
                },
                "style": {
                    "type": "string",
                    "default": "",
                    "description": "可选风格/语气提示（如 gentle、whispering、excited；留空不附加额外标签）。",
                },
                "voice": {
                    "type": "string",
                    "default": "",
                    "description": "可选音色覆盖（本地 voices/ 文件名或云端 reference_id）；留空使用默认音色。",
                },
            },
            "additionalProperties": False,
        },
        generation_prompt=(
            "若本次准备转为语音发送，请保持自然口语与适中篇幅，避免 Markdown 列表、链接或代码块；"
            "可按语境自然使用简短副语言提示（如 [laughing]、[sighs]、[whispering]）。"
        ),
        timeout_ms=90000,
    )
    async def voice_reply_extension(
        self,
        phase: str = "",
        parameters: Optional[dict[str, Any]] = None,
        text: str = "",
        messages: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """MaiSaka 原生回复扩展：在 Planner 选择语音时将回复文本分段合成并替换为 voice 段。"""
        del text, kwargs
        if phase == "prepare":
            # 保持返回空字典以同时兼容 reply 工具串行模式与 passive_replyer 并行补充模式
            return {}
        if phase != "before_send" or not isinstance(messages, list) or not messages:
            return {}

        if not self._is_enabled() or self._auto_voice_mode() != "planner":
            self._log.info(
                "Planner 语音扩展未生效 (enabled=%s, mode=%s)，保留文字发送",
                self._is_enabled(),
                self._auto_voice_mode(),
            )
            cleaned = self._strip_tags_in_reply_messages(messages)
            return {"messages": cleaned} if cleaned is not None else {}

        params = parameters if isinstance(parameters, dict) else {}
        if not bool(params.get("use_voice", True)):
            self._log.info("Planner 语音扩展参数 use_voice=False，保留文字发送")
            cleaned = self._strip_tags_in_reply_messages(messages)
            return {"messages": cleaned} if cleaned is not None else {}

        explicit_request = bool(params.get("explicit_request", False))
        try:
            suitability = float(params.get("suitability", 0.7))
        except (TypeError, ValueError):
            suitability = 0.7

        av = self.config.auto_voice
        if not _should_allow_planner_voice(
            float(av.planner_temperature),
            explicit_request=explicit_request,
            suitability=suitability,
        ):
            self._log.info(
                "Planner 语音扩展被温度阈值过滤 (temp=%.2f, suitability=%.2f, explicit=%s)，保留文字发送",
                float(av.planner_temperature),
                suitability,
                explicit_request,
            )
            cleaned = self._strip_tags_in_reply_messages(messages)
            return {"messages": cleaned} if cleaned is not None else {}

        plan_entries: list[tuple[bool, dict[str, Any], list[str]]] = []
        convertible_texts: list[str] = []
        for orig_msg in messages:
            if not isinstance(orig_msg, dict):
                continue
            raw_segments = orig_msg.get("segments")
            spoken_text = _extract_plain_text_from_segments(raw_segments)
            if not spoken_text:
                plan_entries.append((False, orig_msg, []))
                continue
            convertible_texts.append(spoken_text)
            sub_segments = split_tts_segments(
                spoken_text,
                max_chars=int(av.max_segment_chars),
                split_sentences=bool(av.split_sentences),
            )
            plan_entries.append((bool(sub_segments), orig_msg, sub_segments))

        combined_spoken = "\n".join(convertible_texts).strip()
        if not convertible_texts:
            self._log.info("Planner 语音扩展未检测到可朗读纯文本分段，保留原消息发送")
            cleaned = self._strip_tags_in_reply_messages(messages)
            return {"messages": cleaned} if cleaned is not None else {}
        ineligible_reason = self._explain_ineligible_auto_voice_text(combined_spoken)
        if ineligible_reason is not None:
            self._log.info("Planner 语音扩展跳过转语音 (%s)，保留文字发送", ineligible_reason)
            cleaned = self._strip_tags_in_reply_messages(messages)
            return {"messages": cleaned} if cleaned is not None else {}

        flat_segments = [seg for is_conv, _, segs in plan_entries if is_conv for seg in segs]
        if not flat_segments:
            return {}

        style_arg = str(params.get("style") or "").strip()
        voice_arg = str(params.get("voice") or "").strip()
        synthesized: list[dict[str, Any]] = []
        try:
            synthesized, failed_from, synth_err = await self._synthesize_segments(
                flat_segments,
                voice=voice_arg,
                style=style_arg,
                total_budget_sec=75.0,
            )
            if failed_from is not None and not synthesized:
                self._log.warning("Planner 语音扩展首段合成未完成 (%s)，保留文字发送", synth_err)
                if not self.config.fallback.fallback_to_text:
                    raise RuntimeError(synth_err or "Fish Audio 语音合成失败")
                cleaned = self._strip_tags_in_reply_messages(messages)
                return {"messages": cleaned} if cleaned is not None else {}

            out_messages: list[dict[str, Any]] = []
            flat_idx = 0
            fallback_to_text = bool(self.config.fallback.fallback_to_text)
            for is_conv, orig_msg, sub_segments in plan_entries:
                if not is_conv:
                    out_messages.append(orig_msg)
                    continue
                emitted_fallback = False
                for sub_seg in sub_segments:
                    if flat_idx < len(synthesized):
                        item = synthesized[flat_idx]
                        out_messages.append(
                            {
                                "segments": [
                                    {
                                        "type": "voice",
                                        "data": self._select_voice_display_text(item),
                                        "binary_data_base64": str(item["audio_base64"]),
                                    }
                                ],
                                "quote_previous": False,
                            }
                        )
                    elif fallback_to_text:
                        quote_prev = bool(orig_msg.get("quote_previous", False)) and not emitted_fallback
                        emitted_fallback = True
                        out_messages.append(
                            {
                                "segments": [{"type": "text", "data": self._format_fallback_text(sub_seg)}],
                                "quote_previous": quote_prev,
                            }
                        )
                    flat_idx += 1

            if not out_messages:
                raise RuntimeError(synth_err or "Fish Audio 语音合成失败")

            if self._should_send_injected_preview_to_chat():
                preview_text = self._collect_injected_preview_text(synthesized)
                if preview_text:
                    out_messages.append(
                        {
                            "segments": [{"type": "text", "data": preview_text}],
                            "quote_previous": False,
                        }
                    )

            last_item = synthesized[-1]
            self._log.info(
                "Planner 语音扩展改写完成 (segments=%d/%d, model=%s, voice=%s, temp=%.2f, suitability=%.2f, explicit=%s)",
                len(synthesized),
                len(flat_segments),
                last_item["model"],
                last_item["voice"],
                float(av.planner_temperature),
                suitability,
                explicit_request,
            )
            return {"messages": out_messages}
        finally:
            await self._cleanup_deferred_files(synthesized)

    @HookHandler(
        "send_service.before_send",
        name="fish_auto_voice",
        description="按触发模式（always 全量 / probability 随机概率）将出站纯文本转为 Fish Audio 语音，并保护语音消息不被引用标记阻断",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        timeout_ms=90000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def auto_voice_hook(self, **kwargs: Any) -> Optional[dict[str, Any]]:
        """出站前拦截：支持 always / probability 模式转语音，并对已转语音消息清除不兼容的引用标记。"""
        if not self._is_enabled():
            return None

        message = kwargs.get("message")
        if not isinstance(message, dict):
            return None

        if _has_voice_segment(message.get("raw_message")):
            if kwargs.get("set_reply"):
                updated_kwargs = dict(kwargs)
                updated_kwargs["set_reply"] = False
                updated_kwargs["reply_message_id"] = ""
                return {"action": "continue", "modified_kwargs": updated_kwargs}
            return None

        mode = self._auto_voice_mode()
        if mode not in {"always", "probability"}:
            return None

        av = getattr(self.config, "auto_voice", None)
        if av is None:
            return None

        stream_id = str(kwargs.get("stream_id") or message.get("session_id") or "").strip()
        if not stream_id:
            return None

        spoken_text = _extract_outbound_plain_text(message)
        if not spoken_text:
            return None

        if self._consume_fallback_bypass(stream_id, spoken_text):
            return None

        if not self._is_eligible_auto_voice_text(spoken_text):
            return None

        if mode == "probability":
            should_convert, from_cache = self._should_convert_by_probability(
                stream_id,
                float(av.random_probability),
                float(getattr(av, "decision_ttl_seconds", 8.0) or 0.0),
            )
            if not should_convert:
                self._log.debug(
                    "纯随机语音跳过 (stream=%s, prob=%.2f, cached=%s)",
                    stream_id,
                    float(av.random_probability),
                    from_cache,
                )
                return None

        segments = split_tts_segments(
            spoken_text,
            max_chars=int(av.max_segment_chars),
            split_sentences=bool(av.split_sentences),
        )
        if not segments:
            return None

        delay_ms = int(av.segment_delay_ms)
        synthesized: list[dict[str, Any]] = []
        sent_head = 0
        try:
            synthesized, failed_from, synth_err = await self._synthesize_segments(
                segments,
                total_budget_sec=75.0,
            )
            if failed_from is not None:
                if not synthesized:
                    self._log.warning("出站自动语音首段未完成 (%s)，改由原文字发出", synth_err)
                    return None
                sent_count, send_err = await self._send_segment_sequence(
                    synthesized,
                    stream_id,
                    delay_ms=delay_ms,
                )
                sent_head = sent_count
                remaining_text = "\n".join(segments[sent_count:])
                self._log.warning(
                    "分段语音第 %d/%d 段合成失败 (%s)，已发送前 %d/%d 段语音（发送错误=%s），剩余文字原样放行 (stream=%s)",
                    failed_from + 1,
                    len(segments),
                    synth_err,
                    sent_count,
                    len(synthesized),
                    send_err or "无",
                    stream_id,
                )
                return _build_text_override_hook_result(kwargs, message, remaining_text)

            if len(synthesized) > 1:
                sent_count, head_err = await self._send_segment_sequence(
                    synthesized[:-1],
                    stream_id,
                    delay_ms=delay_ms,
                )
                sent_head = sent_count
                if head_err:
                    await self._cleanup_deferred_files([synthesized[-1]])
                    remaining_text = "\n".join(segments[sent_count:])
                    self._log.warning(
                        "分段语音前序发送中断 (sent=%d/%d, err=%s)，剩余片段改为文字发出",
                        sent_count,
                        len(segments),
                        head_err,
                    )
                    return _build_text_override_hook_result(kwargs, message, remaining_text)
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000.0)

            last_item = synthesized[-1]
            await self._cleanup_deferred_files([last_item])

            self._log.info(
                "出站语音改写完成 (mode=%s, stream=%s, segments=%d, model=%s, voice=%s)",
                mode,
                stream_id,
                len(synthesized),
                last_item["model"],
                last_item["voice"],
            )
            await self._maybe_send_injected_preview_to_chat(synthesized, stream_id)
            return _build_voice_hook_result(
                kwargs,
                message,
                str(last_item["audio_base64"]),
                self._select_voice_display_text(last_item),
            )
        except Exception as exc:  # noqa: BLE001
            self._log.warning("出站语音链路异常中断: %s", exc)
            await self._cleanup_deferred_files(synthesized[sent_head:])
            if sent_head > 0:
                return _build_text_override_hook_result(
                    kwargs,
                    message,
                    "\n".join(segments[sent_head:]),
                )
            return None

    async def _deliver_voice_reply(
        self,
        raw_text: str,
        *,
        stream_id: str,
        voice: str = "",
        style: str = "",
        force_emotion: bool = False,
    ) -> dict[str, Any]:
        """执行文本分段、情绪注入、语音合成、发送及失败降级公共流程。"""
        av = getattr(self.config, "auto_voice", None)
        should_split = bool(av and av.split_sentences) and len(raw_text) > int(av.max_segment_chars)
        segments = (
            split_tts_segments(
                raw_text,
                max_chars=int(av.max_segment_chars),
                split_sentences=True,
            )
            if should_split
            else [raw_text]
        ) or [raw_text]

        delay_ms = int(av.segment_delay_ms) if av is not None else 0
        synthesized, failed_from, synth_err = await self._synthesize_segments(
            segments,
            voice=voice,
            style=style,
            force_emotion=force_emotion,
        )
        if failed_from is not None:
            sent_count = 0
            if synthesized:
                sent_count, _ = await self._send_segment_sequence(
                    synthesized,
                    stream_id,
                    delay_ms=delay_ms,
                )
            remaining = "\n".join(segments[sent_count:])
            fb_res = await self._fallback_to_text(remaining, stream_id, reason=synth_err)
            return {
                "success": False,
                "status": "fallback_text" if fb_res.get("fallback_sent") else "failed",
                "error": synth_err,
                "fallback_sent": fb_res.get("fallback_sent", False),
                "partial_voice_segments": sent_count,
            }

        sent_count, send_err = await self._send_segment_sequence(
            synthesized,
            stream_id,
            delay_ms=delay_ms,
        )
        if send_err:
            remaining = "\n".join(segments[sent_count:])
            fb_res = await self._fallback_to_text(remaining, stream_id, reason=f"发送语音失败: {send_err}")
            return {
                "success": False,
                "status": "fallback_text" if fb_res.get("fallback_sent") else "failed",
                "error": send_err,
                "fallback_sent": fb_res.get("fallback_sent", False),
                "partial_voice_segments": sent_count,
            }

        total_bytes = sum(int(item["audio_bytes_len"]) for item in synthesized)
        last_item = synthesized[-1]
        await self._maybe_send_injected_preview_to_chat(
            synthesized,
            stream_id,
            force_emotion=force_emotion,
        )
        self._log.info(
            "语音回复发送成功 (stream=%s, segments=%d, model=%s, voice=%s, bytes=%d, fmt=%s, force_emotion=%s)",
            stream_id,
            len(synthesized),
            last_item["model"],
            last_item["voice"],
            total_bytes,
            last_item["format"],
            force_emotion,
        )
        return {
            "success": True,
            "status": "sent",
            "content": "语音已合成并发送给对方，无需再用 reply 发送重复文字。",
            "stream_id": stream_id,
            "segments": len(synthesized),
            "model": last_item["model"],
            "voice": last_item["voice"],
            "format": last_item["format"],
            "audio_bytes_len": total_bytes,
        }

    @staticmethod
    def _parse_fishaudio_command_args(kwargs: dict[str, Any]) -> tuple[str, bool]:
        """解析 /fishaudio 命令参数，返回 (待合成文本, 是否使用 emo 强制注入情绪)。"""
        raw_arg = ""
        matched = kwargs.get("matched_groups")
        if isinstance(matched, dict):
            raw_arg = str(matched.get("text") or "").strip()
        if not raw_arg:
            raw_msg = str(kwargs.get("text") or "").strip()
            cmd_match = re.match(r"(?is)^/fishaudio(?:\s+(?P<text>.+))?\s*$", raw_msg)
            if cmd_match:
                raw_arg = str(cmd_match.group("text") or "").strip()
        if not raw_arg:
            return "", False

        if raw_arg.lower() == "emo":
            return "", True
        emo_match = re.match(r"^emo\s+(?P<body>.+)$", raw_arg, re.IGNORECASE | re.DOTALL)
        if emo_match:
            return str(emo_match.group("body") or "").strip(), True
        return raw_arg, False

    @Command(
        "fishaudio",
        description="强制将后续文本转换为 Fish Audio 语音；使用 /fishaudio emo <文本> 可强制注入情绪标签",
        pattern=r"(?is)^/fishaudio(?:\s+(?P<text>.+))?\s*$",
        timeout_ms=120000,
    )
    async def handle_fishaudio_command(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """处理 /fishaudio [emo] <文本> 命令，强制转换为语音并拦截后续默认回复。"""
        stream_id = await self._find_stream_id(kwargs)
        if not self._is_enabled():
            if stream_id:
                disabled_tip = "Fish Audio 语音插件当前已禁用，请在配置中开启。"
                self._register_fallback_bypass(stream_id, disabled_tip)
                await self.ctx.send.text(disabled_tip, stream_id)
            return False, "Fish Audio 语音插件当前已禁用", True

        target_text, force_emotion = self._parse_fishaudio_command_args(kwargs)
        if not target_text:
            if stream_id:
                usage_tip = (
                    "用法：/fishaudio <要转换为语音的文本>\n"
                    "强制注入情绪标签：/fishaudio emo <要转换为语音的文本>"
                )
                self._register_fallback_bypass(stream_id, usage_tip)
                await self.ctx.send.text(usage_tip, stream_id)
            return False, "缺少待转换的文本内容", True

        if not stream_id:
            self._log.warning("/fishaudio 命令未找到目标 stream_id，放弃语音合成以避免跨群错发")
            return False, "无法定位当前聊天流 stream_id", True

        res = await self._deliver_voice_reply(
            target_text,
            stream_id=stream_id,
            force_emotion=force_emotion,
        )
        if res.get("success"):
            return True, f"已发送语音（共 {int(res.get('segments') or 1)} 段）", True
        return False, str(res.get("error") or "语音转换失败"), True

    @Tool(
        "send_voice_reply",
        description=(
            "使用 Fish Audio TTS 将指定台词合成为语音并直接发送到当前会话。"
            "优先建议通过 reply 工具的 plugin_options 启用语音扩展以保留完整人设回复；"
            "当需要独立直接发送一段短语音（如撒娇、慰问、晚安或对方明确要听语音）时可调用本工具，"
            "调用成功后不要再用 reply 发送重复文字。可在文本中自然插入 [laughing]、[sighs]、[whispering] 等标签。"
        ),
        visibility="visible",
        parameters=[
            ToolParameterInfo(
                name="text",
                param_type=ToolParamType.STRING,
                description="要合成语音并发送的台词正文",
                required=True,
            ),
            ToolParameterInfo(
                name="voice",
                param_type=ToolParamType.STRING,
                description="可选音色覆盖：本地 voices/ 文件名（不含扩展名）或 Fish Audio 云端 reference_id；留空则使用默认配置",
                required=False,
                default="",
            ),
            ToolParameterInfo(
                name="style",
                param_type=ToolParamType.STRING,
                description="可选风格/语气提示（如 gentle、whispering、excited；留空不附加额外标签）",
                required=False,
                default="",
            ),
        ],
    )
    async def send_voice_reply(
        self,
        text: str,
        voice: str = "",
        style: str = "",
        *,
        force_emotion: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """合成语音并发送至当前聊天流，支持自动分段，失败时按配置自动退回纯文字。"""
        if not self._is_enabled():
            return {"success": False, "status": "disabled", "message": "Fish Audio 语音插件当前已禁用"}

        raw_text = (text or "").strip()
        stream_id = await self._find_stream_id(kwargs)

        if not raw_text:
            return await self._fallback_to_text(raw_text, stream_id, reason="台词内容为空")

        if not stream_id:
            self._log.warning("send_voice_reply 未找到目标 stream_id，放弃语音合成以避免跨群错发")
            return {"success": False, "status": "no_stream", "error": "无法定位当前聊天流 stream_id"}

        return await self._deliver_voice_reply(
            raw_text,
            stream_id=stream_id,
            voice=voice,
            style=style,
            force_emotion=force_emotion,
        )

    @API(
        "fish_tts",
        description="供其他插件或外部流程直接调用的 Fish Audio TTS 合成接口，返回 Base64 音频数据",
        version="1",
        public=True,
    )
    async def api_synthesize(
        self,
        text: str,
        voice: str = "",
        style: str = "",
        audio_format: str = "",
        speed: Optional[float] = None,
        wrap_pcm: bool = True,
    ) -> dict[str, Any]:
        """跨插件公共合成 API。"""
        return await self._synthesize_pipeline(
            text,
            voice=voice,
            style=style,
            audio_format=audio_format,
            speed=speed,
            wrap_pcm=wrap_pcm,
        )


def create_plugin() -> FishAudioVoicePlugin:
    """MaiBot SDK 插件工厂入口。"""
    return FishAudioVoicePlugin()
