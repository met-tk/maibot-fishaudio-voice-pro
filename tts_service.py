"""Fish Audio TTS 服务模块（适配 POST /v1/tts OpenAPI）。"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import aiohttp

try:
    import msgpack
except ImportError:  # pragma: no cover - 宿主环境已内置 msgpack，仅作降级保护
    msgpack = None


def resolve_tts_endpoint(raw_url: str) -> str:
    """将基础地址或完整路径归一化为 Fish Audio `/v1/tts` 端点。"""
    cleaned = (raw_url or "").strip().rstrip("/")
    if not cleaned:
        return FishAudioTTSService.DEFAULT_API_URL
    lowered = cleaned.lower()
    if lowered.endswith("/v1/tts"):
        return cleaned
    if lowered.endswith("/v1"):
        return f"{cleaned}/tts"
    return f"{cleaned}/v1/tts"


@dataclass(frozen=True)
class FishSynthesisOptions:
    """Fish Audio `/v1/tts` 合成参数快照。"""

    model: str = "s2.1-pro-free"
    audio_format: str = "mp3"
    sample_rate: int = 0
    mp3_bitrate: int = 128
    opus_bitrate: int = -1000
    latency: str = "balanced"
    temperature: float = 0.7
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    chunk_length: int = 300
    min_chunk_length: int = 50
    normalize: bool = True
    language: str = ""
    condition_on_previous_chunks: bool = True
    max_new_tokens: int = 1024
    early_stop_threshold: float = 1.0
    prosody_speed: float = 1.0
    prosody_volume: float = 0.0
    prosody_normalize_loudness: bool = True


@dataclass(frozen=True)
class ReferenceSample:
    """即时复刻参考音频样本（音频二进制 + 台词文本）。"""

    audio: bytes
    text: str = ""


class FishAudioTTSService:
    """Fish Audio TTS 异步客户端。

    设计对齐高可用语音回复链路：
    1. 区分建连超时、传输空闲超时（sock_read）与整体超时，避免长语音在持续传输时被误杀；
    2. 仅对瞬时状态码（408/425/429/5xx）与网络异常指数退避重试，超时及 401/402/422 不重试；
    3. 使用信号量限制全局并发请求数；
    4. 同时支持 JSON（默认音色 / 云端 reference_id）与 MessagePack（本地 references 即时复刻）。
    """

    DEFAULT_API_URL = "https://api.fish.audio/v1/tts"
    DEFAULT_MODEL = "s2.1-pro-free"
    SUPPORTED_MODELS = ("s2.1-pro-free", "s2.1-pro", "s2-pro", "drama-3", "s1")
    SUPPORTED_FORMATS = ("mp3", "wav", "pcm", "opus")
    SUPPORTED_LATENCIES = ("balanced", "normal", "low")
    SUPPORTED_MP3_BITRATES = (64, 128, 192)
    SUPPORTED_OPUS_BITRATES = (-1000, 24000, 32000, 48000, 64000)

    DEFAULT_REQUEST_TIMEOUT = 90.0
    DEFAULT_CONNECT_TIMEOUT = 10.0
    DEFAULT_STALL_TIMEOUT = 30.0
    DEFAULT_MAX_RETRIES = 2
    DEFAULT_RETRY_BACKOFF = 1.5
    DEFAULT_MAX_CONCURRENCY = 1

    RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

    def __init__(
        self,
        api_key: str,
        api_base_url: str = "",
        logger: Optional[logging.Logger] = None,
        *,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        stall_timeout: Optional[float] = None,
    ) -> None:
        self.api_key = api_key.strip() if api_key else ""
        self.api_base_url = resolve_tts_endpoint(api_base_url)
        self.logger = logger or logging.getLogger(__name__)
        self._session: Optional[aiohttp.ClientSession] = None
        self._retired_sessions: list[Any] = []
        self._inflight_requests: int = 0
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._max_concurrency = 1
        self._update_limits(
            request_timeout=request_timeout,
            connect_timeout=connect_timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            max_concurrency=max_concurrency,
            stall_timeout=stall_timeout,
        )

    def _update_limits(
        self,
        *,
        request_timeout: float,
        connect_timeout: float,
        max_retries: int,
        retry_backoff: float,
        max_concurrency: int,
        stall_timeout: Optional[float] = None,
    ) -> None:
        self.request_timeout = max(1.0, float(request_timeout))
        self.stall_timeout = max(
            1.0, float(stall_timeout) if stall_timeout is not None else self.request_timeout
        )
        self.connect_timeout = max(1.0, float(connect_timeout))
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        new_limit = max(1, int(max_concurrency))
        if self._max_concurrency != new_limit or self._semaphore is None:
            self._max_concurrency = new_limit
            self._semaphore = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
        return self._semaphore

    def _build_timeout(self) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(
            total=self.request_timeout,
            sock_connect=self.connect_timeout,
            sock_read=self.stall_timeout,
        )

    def build_headers(self, model: str, *, use_msgpack: bool) -> dict[str, str]:
        """构造 Fish Audio `/v1/tts` HTTP 请求头。"""
        normalized_model = (model or "").strip() or self.DEFAULT_MODEL
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/msgpack" if use_msgpack else "application/json",
            "model": normalized_model,
        }

    def build_payload(
        self,
        text: str,
        options: FishSynthesisOptions,
        *,
        reference_id: str | Sequence[str] = "",
        references: Optional[Sequence[ReferenceSample]] = None,
    ) -> dict[str, Any]:
        """按 Fish Audio OpenAPI `TTSRequest` 规范构造请求体。"""
        fmt = (options.audio_format or "mp3").strip().lower()
        if fmt not in self.SUPPORTED_FORMATS:
            fmt = "mp3"

        latency = (options.latency or "balanced").strip().lower()
        if latency not in self.SUPPORTED_LATENCIES:
            latency = "balanced"

        payload: dict[str, Any] = {
            "text": text,
            "format": fmt,
            "latency": latency,
            "temperature": max(0.0, min(float(options.temperature), 1.0)),
            "top_p": max(0.0, min(float(options.top_p), 1.0)),
            "repetition_penalty": max(0.5, min(float(options.repetition_penalty), 2.5)),
            "chunk_length": max(100, min(int(options.chunk_length), 300)),
            "min_chunk_length": max(0, min(int(options.min_chunk_length), 100)),
            "normalize": bool(options.normalize),
            "condition_on_previous_chunks": bool(options.condition_on_previous_chunks),
            "max_new_tokens": max(64, min(int(options.max_new_tokens), 4096)),
            "early_stop_threshold": max(0.0, min(float(options.early_stop_threshold), 1.0)),
            "prosody": {
                "speed": max(0.5, min(float(options.prosody_speed), 2.0)),
                "volume": max(-20.0, min(float(options.prosody_volume), 20.0)),
                "normalize_loudness": bool(options.prosody_normalize_loudness),
            },
        }

        lang = (options.language or "").strip()
        if lang and lang.lower() != "auto":
            payload["language"] = lang

        if options.sample_rate and int(options.sample_rate) > 0:
            payload["sample_rate"] = int(options.sample_rate)

        if fmt == "mp3":
            bitrate = int(options.mp3_bitrate)
            payload["mp3_bitrate"] = bitrate if bitrate in self.SUPPORTED_MP3_BITRATES else 128
        elif fmt == "opus":
            bitrate = int(options.opus_bitrate)
            payload["opus_bitrate"] = bitrate if bitrate in self.SUPPORTED_OPUS_BITRATES else -1000

        if references:
            payload["references"] = [
                {"audio": bytes(sample.audio), "text": str(sample.text or "")}
                for sample in references
            ]
        elif isinstance(reference_id, str):
            rid_parts = [part.strip() for part in reference_id.split(",") if part.strip()]
            if len(rid_parts) == 1:
                payload["reference_id"] = rid_parts[0]
            elif len(rid_parts) > 1:
                payload["reference_id"] = rid_parts
        elif reference_id:
            rid_list = [str(item).strip() for item in reference_id if str(item).strip()]
            if len(rid_list) == 1:
                payload["reference_id"] = rid_list[0]
            elif len(rid_list) > 1:
                payload["reference_id"] = rid_list

        return payload

    @staticmethod
    def encode_body(payload: dict[str, Any], *, use_msgpack: bool) -> bytes:
        """将请求体序列化为 JSON 或 MessagePack 字节。"""
        if use_msgpack:
            if msgpack is None:
                raise RuntimeError("msgpack 未安装，无法使用本地音频即时复刻（clone）模式")
            return bytes(msgpack.packb(payload, use_bin_type=True))
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    async def synthesize(
        self,
        text: str,
        *,
        options: Optional[FishSynthesisOptions] = None,
        reference_id: str | Sequence[str] = "",
        references: Optional[Sequence[ReferenceSample]] = None,
    ) -> dict[str, Any]:
        """执行 Fish Audio TTS 合成。"""
        if not self.api_key:
            return {"success": False, "error": "Fish Audio API Key not configured"}

        if not text or not text.strip():
            return {"success": False, "error": "Empty text"}

        opts = options or FishSynthesisOptions()
        use_msgpack = bool(references)
        try:
            payload = self.build_payload(
                text=text,
                options=opts,
                reference_id=reference_id,
                references=references,
            )
            body_bytes = self.encode_body(payload, use_msgpack=use_msgpack)
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        headers = self.build_headers(opts.model, use_msgpack=use_msgpack)
        key_preview = f"{self.api_key[:4]}****" if len(self.api_key) > 4 else "****"
        mode_desc = f"clone(refs={len(references or ())})" if use_msgpack else (
            f"reference_id={payload.get('reference_id')}" if "reference_id" in payload else "default"
        )
        self.logger.info(
            "Fish Audio TTS: url=%s, key=%s, model=%s, mode=%s, format=%s, body_bytes=%d, text=%s",
            self.api_base_url,
            key_preview,
            headers["model"],
            mode_desc,
            payload["format"],
            len(body_bytes),
            text,
        )
        return await self._do_request(body_bytes=body_bytes, headers=headers, text_content=text)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ttl_dns_cache=10)
            self._session = aiohttp.ClientSession(
                connector=connector,
                trust_env=True,
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=self.connect_timeout,
                ),
            )
        return self._session

    async def close(self) -> None:
        sessions: list[Any] = []
        if self._session is not None:
            sessions.append(self._session)
            self._session = None
        if self._retired_sessions:
            sessions.extend(self._retired_sessions)
            self._retired_sessions.clear()
        for session in sessions:
            if session is not None and not getattr(session, "closed", True):
                await session.close()

    async def _flush_retired_sessions(self) -> None:
        if self._inflight_requests > 0 or not self._retired_sessions:
            return
        pending = list(self._retired_sessions)
        self._retired_sessions.clear()
        for session in pending:
            if session is not None and not getattr(session, "closed", True):
                await session.close()

    async def _discard_session(self, session: Any) -> None:
        """发生网络或建连异常时替换陈旧连接池；存在在途并发请求时延后安全关闭。"""
        if session is None:
            return
        if self._session is session:
            self._session = None
        if getattr(session, "closed", True):
            return
        if self._inflight_requests <= 0:
            await session.close()
        elif session not in self._retired_sessions:
            self._retired_sessions.append(session)

    @staticmethod
    def _format_client_error(exc: Exception) -> str:
        detail = str(exc).strip()
        os_error = getattr(exc, "os_error", None)
        if os_error is not None:
            errno = getattr(os_error, "winerror", None) or getattr(os_error, "errno", None)
            if "[None]" in detail or not detail:
                return f"{type(exc).__name__}: {detail or repr(exc)} (os_error={os_error!r}, errno={errno})"
        return detail or repr(exc)

    async def _do_request(
        self,
        *,
        body_bytes: bytes,
        headers: dict[str, str],
        text_content: str,
    ) -> dict[str, Any]:
        attempts = self.max_retries + 1
        last_error = "unknown"

        for attempt in range(attempts):
            if attempt:
                delay = self.retry_backoff * (2 ** (attempt - 1))
                self.logger.info(
                    "Fish Audio TTS 第 %d/%d 次重试，等待 %.1fs（上次：%s）",
                    attempt + 1,
                    attempts,
                    delay,
                    last_error,
                )
                await asyncio.sleep(delay)

            session: Any = None
            try:
                async with self._get_semaphore():
                    session = await self._get_session()
                    self._inflight_requests += 1
                    try:
                        async with session.post(
                            self.api_base_url,
                            data=body_bytes,
                            headers=headers,
                            timeout=self._build_timeout(),
                        ) as response:
                            if response.status == 200:
                                return await self._parse_response(response, text_content)

                            error_body = await response.read()
                            last_error = self._format_http_error(response.status, error_body)
                            self.logger.error("Fish Audio TTS API failed: %s", last_error)
                            if response.status not in self.RETRYABLE_STATUS:
                                return {"success": False, "error": last_error}
                            continue
                    finally:
                        self._inflight_requests = max(0, self._inflight_requests - 1)

            except aiohttp.ConnectionTimeoutError as exc:
                last_error = f"Connect timeout ({self.connect_timeout}s): {self._format_client_error(exc)}"
                self.logger.warning(
                    "Fish Audio TTS 建连超时 (%d/%d): %s",
                    attempt + 1,
                    attempts,
                    last_error,
                )
                await self._discard_session(session)
                continue
            except (asyncio.TimeoutError, TimeoutError):
                await self._discard_session(session)
                self.logger.warning(
                    "Fish Audio TTS 超时（整体 %ss / 传输空闲 %ss），放弃本次语音合成",
                    self.request_timeout,
                    self.stall_timeout,
                )
                return {"success": False, "error": self._timeout_error_text()}
            except aiohttp.ClientError as exc:
                last_error = f"Network error: {self._format_client_error(exc)}"
                self.logger.error("Fish Audio TTS network error: %s", last_error)
                await self._discard_session(session)
                continue
            except Exception as exc:
                self.logger.error("Fish Audio TTS error: %s", exc)
                return {"success": False, "error": str(exc)}
            finally:
                await self._flush_retired_sessions()

        return {"success": False, "error": last_error}

    def _timeout_error_text(self) -> str:
        return (
            f"timeout after {self.request_timeout}s (total) / "
            f"{self.stall_timeout}s (idle)"
        )

    @staticmethod
    def _format_http_error(status: int, error_body: bytes) -> str:
        raw_text = error_body.decode("utf-8", "ignore").strip()
        if raw_text.startswith("{"):
            try:
                parsed = json.loads(raw_text)
                if isinstance(parsed, dict):
                    msg = str(parsed.get("message") or parsed.get("detail") or "").strip()
                    reason = str(parsed.get("reason") or "").strip()
                    if msg or reason:
                        detail = f"{msg} ({reason})" if msg and reason else (msg or reason)
                        return f"API failed: {status} - {detail[:200]}"
            except Exception:
                pass
        return f"API failed: {status} - {raw_text[:200]}"

    async def _parse_response(
        self,
        response: aiohttp.ClientResponse,
        text_content: str,
    ) -> dict[str, Any]:
        try:
            body = await response.read()
        except (asyncio.TimeoutError, TimeoutError):
            self.logger.warning(
                "Fish Audio TTS 读取响应体超时（整体 %ss / 传输空闲 %ss）",
                self.request_timeout,
                self.stall_timeout,
            )
            return {"success": False, "error": self._timeout_error_text()}
        except Exception as exc:
            self.logger.error("Fish Audio TTS 读取响应体失败: %s: %s", type(exc).__name__, exc)
            return {"success": False, "error": f"read body failed: {type(exc).__name__}: {exc}"}

        if not body:
            self.logger.error("Fish Audio TTS 响应体为空")
            return {"success": False, "error": "empty response body"}

        content_type = ""
        if getattr(response, "headers", None):
            content_type = str(response.headers.get("Content-Type", "")).lower()

        if "application/json" in content_type or body.lstrip().startswith(b"{"):
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    err = self._format_http_error(200, body)
                    self.logger.error("Fish Audio TTS 返回了 JSON 而非音频: %s", err)
                    return {"success": False, "error": err}
            except Exception:
                if "application/json" in content_type:
                    return {"success": False, "error": f"unexpected json response: {body[:200]!r}"}

        audio_b64 = base64.b64encode(body).decode("ascii")
        return {
            "success": True,
            "audio_base64": audio_b64,
            "audio_bytes_len": len(body),
            "text": text_content,
        }
