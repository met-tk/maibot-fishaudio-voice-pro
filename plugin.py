"""Fish Audio TTS 与音色克隆 MaiBot 插件。

提供基于 Fish Audio OpenAPI v1 (POST /v1/tts) 的语音合成与即时音色复刻能力，
支持 s2.1-pro-free / s2.1-pro / s2-pro / drama-3 / s1 模型，支持云端 reference_id
以及本地 voices/ 目录参考音频 MessagePack 即时克隆，并在失败时自动降级为纯文本回复。
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

from maibot_sdk import (
    API,
    CONFIG_RELOAD_SCOPE_SELF,
    Field,
    HookHandler,
    MaiBotPlugin,
    PluginConfigBase,
    Tool,
)
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder, ToolParameterInfo, ToolParamType

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
        default="1.0.0",
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
    """全局默认文字转语音与分段发送配置。"""

    __ui_label__ = "全局默认语音与分段"

    enabled: bool = Field(
        default=True,
        description="开启后默认将所有符合条件的出站纯文本回复自动转为语音发送",
        json_schema_extra={"label": "默认启用语音回复", "order": 1},
    )
    split_sentences: bool = Field(
        default=True,
        description="遇到多句或较长文本时按句拆分为多条语音依次分段发送",
        json_schema_extra={"label": "长文本分段发送", "order": 2},
    )
    max_segment_chars: int = Field(
        default=80,
        ge=15,
        le=300,
        description="分段发送时单条语音允许的最大字符数（超出则按标点继续拆段）",
        json_schema_extra={"label": "单段最大字数", "order": 3},
    )
    segment_delay_ms: int = Field(
        default=200,
        ge=0,
        le=3000,
        description="连续发送多段语音时每段之间的间隔毫秒数（0 表示无额外等待）",
        json_schema_extra={"label": "分段发送间隔 (ms)", "order": 4},
    )
    min_text_length: int = Field(
        default=1,
        ge=1,
        le=50,
        description="自动转为语音的最短文本字符数",
        json_schema_extra={"label": "最小转换字数", "order": 5},
    )
    max_text_length: int = Field(
        default=500,
        ge=20,
        le=4000,
        description="允许自动转语音的单条原始消息最大总字数（超出则保持纯文本避免超时）",
        json_schema_extra={"label": "最大转换字数", "order": 6},
    )
    skip_urls_and_code: bool = Field(
        default=True,
        description="消息正文包含网址链接或代码块时保持纯文本不转语音",
        json_schema_extra={"label": "跳过链接与代码块", "order": 7},
    )


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


class FishAudioPluginConfig(PluginConfigBase):
    """Fish Audio TTS 插件顶层配置模型。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    voice: VoiceSectionConfig = Field(default_factory=VoiceSectionConfig)
    auto_voice: AutoVoiceSectionConfig = Field(default_factory=AutoVoiceSectionConfig)
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


def _pack_fragments(pieces: list[str], limit: int) -> list[str]:
    """将细粒片段按最大长度合并打包，超长单片再做硬切分。"""
    packed: list[str] = []
    current = ""
    for raw_piece in pieces:
        piece = raw_piece.strip()
        if not piece:
            continue
        if len(piece) > limit:
            if current:
                packed.append(current)
                current = ""
            for idx in range(0, len(piece), limit):
                chunk = piece[idx : idx + limit].strip()
                if chunk:
                    packed.append(chunk)
            continue
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= limit:
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
        if len(sentence) <= limit:
            if _SPEAKABLE_CHAR_RE.search(sentence):
                segments.append(sentence)
            continue
        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(sentence) if c.strip()]
        for packed in _pack_fragments(clauses or [sentence], limit):
            if _SPEAKABLE_CHAR_RE.search(packed):
                segments.append(packed)

    return segments


def _collect_at_names(message: dict[str, Any]) -> list[str]:
    """提取出站消息中 @ 段对应的昵称列表，避免正文朗读冗余 @ 称呼。"""
    names: list[str] = []
    for item in message.get("raw_message") or []:
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


def _extract_outbound_plain_text(message: dict[str, Any]) -> Optional[str]:
    """提取可转为语音的出站纯文本；遇到语音/图片/文件/视频等非文本段返回 None。"""
    raw_message = message.get("raw_message")
    if not isinstance(raw_message, list) or not raw_message:
        return None

    parts: list[str] = []
    for item in raw_message:
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

    for name in _collect_at_names(message):
        mention = f"@{name}"
        if mention in text:
            text = text.replace(mention, " ")
    normalized = _MULTI_SPACE_RE.sub(" ", text).strip()
    return normalized or None


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
            "Fish Audio TTS 插件已加载 (enabled=%s, auto_voice=%s, model=%s, voice_mode=%s, has_key=%s, local_voices=%d)",
            self._is_enabled(),
            bool(cfg.auto_voice.enabled),
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
    ) -> None:
        if scope != CONFIG_RELOAD_SCOPE_SELF:
            return
        old_service = self._rebuild_runtime()
        if old_service is not None:
            await old_service.close()
        await self._run_storage_prune()
        self._log.info(
            "Fish Audio TTS 配置已热重载 (enabled=%s, auto_voice=%s, model=%s, voice_mode=%s)",
            self._is_enabled(),
            bool(self.config.auto_voice.enabled),
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
        styled_text = apply_style_instruction(raw_text, style, options.model)
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

        text_to_send = strip_audio_tags(raw_text) if fb.strip_audio_tags else (raw_text or "").strip()
        if not text_to_send:
            text_to_send = str(fb.empty_text_placeholder or "（想说点什么，但声音没发出来）").strip()

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
    ) -> tuple[list[dict[str, Any]], Optional[int], str]:
        """按顺序合成多个文本分片，受可选总预算约束，返回 (已成功列表, 首个失败索引, 失败原因)。"""
        synthesized: list[dict[str, Any]] = []
        deadline = time.monotonic() + total_budget_sec if total_budget_sec else None
        for idx, seg_text in enumerate(segments):
            coro = self._synthesize_pipeline(
                seg_text,
                voice=voice,
                style=style,
                wrap_pcm=True,
                defer_immediate_cleanup=True,
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
        "send_service.before_send",
        name="fish_auto_voice",
        description="按配置默认将所有符合条件的出站文字回复转为 Fish Audio 语音并分段发送",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        timeout_ms=90000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def auto_voice_hook(self, **kwargs: Any) -> Optional[dict[str, Any]]:
        """出站前拦截：开启默认语音回复后，将纯文本转为语音并支持自动分段。"""
        if not self._is_enabled():
            return None

        av = getattr(self.config, "auto_voice", None)
        if av is None or not av.enabled:
            return None

        message = kwargs.get("message")
        if not isinstance(message, dict):
            return None

        stream_id = str(kwargs.get("stream_id") or message.get("session_id") or "").strip()
        if not stream_id:
            return None

        spoken_text = _extract_outbound_plain_text(message)
        if not spoken_text:
            return None

        if self._consume_fallback_bypass(stream_id, spoken_text):
            return None

        if len(spoken_text) < int(av.min_text_length) or len(spoken_text) > int(av.max_text_length):
            return None

        if av.skip_urls_and_code and any(pat in spoken_text for pat in ("http://", "https://", "```", "![", "|--")):
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
                    self._log.warning("默认出站语音首段未完成 (%s)，改由原文字发出", synth_err)
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
                "默认出站语音改写完成 (stream=%s, segments=%d, model=%s, voice=%s)",
                stream_id,
                len(synthesized),
                last_item["model"],
                last_item["voice"],
            )
            return _build_voice_hook_result(
                kwargs,
                message,
                str(last_item["audio_base64"]),
                str(last_item["text"]),
            )
        except Exception as exc:  # noqa: BLE001
            self._log.warning("默认出站语音链路异常中断: %s", exc)
            await self._cleanup_deferred_files(synthesized[sent_head:])
            if sent_head > 0:
                return _build_text_override_hook_result(
                    kwargs,
                    message,
                    "\n".join(segments[sent_head:]),
                )
            return None

    @Tool(
        "send_voice_reply",
        description=(
            "使用 Fish Audio TTS 将回复内容合成为语音并发送给当前用户/群聊。"
            "可在文本中自然使用 Fish Audio S2 情感标签（如 [laughing]、[sighs]、[whispering]）增强表达；"
            "若已启用默认全局语音回复，普通文本回复也会自动转为语音发送。"
        ),
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
        self._log.info(
            "语音回复发送成功 (stream=%s, segments=%d, model=%s, voice=%s, bytes=%d, fmt=%s)",
            stream_id,
            len(synthesized),
            last_item["model"],
            last_item["voice"],
            total_bytes,
            last_item["format"],
        )
        return {
            "success": True,
            "status": "sent",
            "stream_id": stream_id,
            "segments": len(synthesized),
            "model": last_item["model"],
            "voice": last_item["voice"],
            "format": last_item["format"],
            "audio_bytes_len": total_bytes,
        }

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
