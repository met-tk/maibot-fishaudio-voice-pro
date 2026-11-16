"""本地音频后处理与裸 PCM 封装（依赖 ffmpeg）。

设计原则：
- 全部默认关闭。未启用任何调音项时 build_filter_chain() 返回空串，直接跳过且不探测 ffmpeg。
- 失败不致命。ffmpeg 不可用、滤镜报错或超时时原样返回原始音频，仅记录 warning。
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_ENCODERS: dict[str, list[str]] = {
    "mp3": ["-c:a", "libmp3lame", "-b:a", "128k"],
    "wav": ["-c:a", "pcm_s16le"],
    "opus": ["-c:a", "libopus", "-b:a", "32k"],
}

# Fish Audio OpenAPI 文档声明 WAV / PCM 默认采样率为 44100 Hz（16-bit 单声道），
# 并支持显式配置 8000 / 16000 / 24000 / 32000 / 44100。
DEFAULT_PCM_SAMPLE_RATE = 44100
RAW_PCM_CHANNELS = 1
RAW_PCM_SAMPLE_WIDTH = 2
RAW_PCM_FORMATS: frozenset[str] = frozenset({"pcm", "pcm16"})

_SEMITONE_RATIO = 2.0 ** (1.0 / 12.0)


def wrap_raw_pcm_to_wav(
    pcm: bytes,
    sample_rate: int = DEFAULT_PCM_SAMPLE_RATE,
    channels: int = RAW_PCM_CHANNELS,
    sample_width: int = RAW_PCM_SAMPLE_WIDTH,
) -> bytes:
    """给无容器头的裸 PCM 字节拼装标准 44 字节 RIFF/WAVE 头。"""
    if not pcm:
        return b""

    rate = int(sample_rate) if sample_rate and int(sample_rate) > 0 else DEFAULT_PCM_SAMPLE_RATE
    block_align = channels * sample_width
    byte_rate = rate * block_align
    data_size = len(pcm)
    riff_size = min(36 + data_size, 0xFFFFFFFF)

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        rate,
        byte_rate,
        block_align,
        sample_width * 8,
        b"data",
        data_size,
    )
    return header + pcm


def _kill_and_reap(proc: asyncio.subprocess.Process) -> None:
    """在任务被取消时立即杀掉 ffmpeg 子进程并异步回收。"""
    if proc.returncode is not None:
        return
    try:
        proc.kill()
    except ProcessLookupError:
        return
    except Exception:  # noqa: BLE001
        return
    asyncio.ensure_future(_wait_quietly(proc))


async def _wait_quietly(proc: asyncio.subprocess.Process) -> None:
    try:
        await proc.wait()
    except Exception:  # noqa: BLE001
        pass


@dataclass(frozen=True)
class AudioOptions:
    """本地音频后处理选项快照。"""

    ffmpeg_path: str = "ffmpeg"
    pitch_semitones: float = 0.0
    preserve_formant: bool = True
    speed: float = 1.0
    loudness_normalize: bool = False
    loudness_target: float = -16.0
    highpass_hz: int = 0
    volume_db: float = 0.0
    fade_ms: int = 0
    trim_silence: bool = False

    @property
    def is_noop(self) -> bool:
        return not (
            abs(self.pitch_semitones) > 1e-6
            or abs(self.speed - 1.0) > 1e-6
            or self.loudness_normalize
            or self.highpass_hz > 0
            or abs(self.volume_db) > 1e-6
            or self.fade_ms > 0
            or self.trim_silence
        )


class AudioProcessor:
    """封装 ffmpeg 音频滤镜链执行器。"""

    def __init__(
        self,
        options: AudioOptions,
        logger: Optional[logging.Logger] = None,
        work_dir: Optional[Path] = None,
    ) -> None:
        self.options = options
        self.logger = logger or logging.getLogger(__name__)
        self.work_dir = work_dir
        self._ffmpeg: Optional[str] = None
        self._resolved = False
        self._warned_unavailable = False

    def _make_tmpdir(self) -> str:
        if self.work_dir is not None:
            try:
                self.work_dir.mkdir(parents=True, exist_ok=True)
                return tempfile.mkdtemp(prefix="fishaudio_audio_", dir=str(self.work_dir))
            except OSError as exc:
                self.logger.debug("runtime_dir 不可用（%r），退回系统临时目录", exc)
        return tempfile.mkdtemp(prefix="fishaudio_audio_")

    def _resolve_ffmpeg(self) -> Optional[str]:
        if self._resolved:
            return self._ffmpeg

        self._resolved = True
        configured = (self.options.ffmpeg_path or "ffmpeg").strip()
        candidates: list[str] = []

        looks_like_path = any(sep in configured for sep in (os.sep, "/", "\\")) or Path(configured).is_absolute()
        configured_missing = False
        if looks_like_path:
            expanded = Path(configured).expanduser()
            if expanded.is_file():
                candidates.append(str(expanded))
            else:
                configured_missing = True
                self.logger.warning("配置的 ffmpeg 路径不存在: %s，将在 PATH 里找", configured)

        found = shutil.which(configured)
        if found:
            candidates.append(found)

        if not candidates:
            for alt in ("ffmpeg", "ffmpeg.exe"):
                if alt == configured:
                    continue
                alt_found = shutil.which(alt)
                if alt_found:
                    candidates.append(alt_found)
                    break

        self._ffmpeg = candidates[0] if candidates else None
        if self._ffmpeg and configured_missing:
            self.logger.warning(
                "audio.ffmpeg_path 配置的 %r 不存在，实际改用 PATH 里的: %s",
                configured,
                self._ffmpeg,
            )
        elif not self._ffmpeg and not self._warned_unavailable:
            self._warned_unavailable = True
            self.logger.warning(
                "已配置音频处理但找不到 ffmpeg（当前路径: %r），本次将跳过音频处理。",
                configured,
            )
        elif self._ffmpeg:
            self.logger.debug("音频处理使用 ffmpeg: %s", self._ffmpeg)
        return self._ffmpeg

    def build_filter_chain(self) -> str:
        if self.options.is_noop:
            return ""

        o = self.options
        filters: list[str] = []

        if o.highpass_hz > 0:
            hz = max(20, min(int(o.highpass_hz), 20000))
            filters.append(f"highpass=f={hz}")

        if abs(o.pitch_semitones) > 1e-6:
            ratio = _SEMITONE_RATIO ** float(o.pitch_semitones)
            ratio = max(0.25, min(ratio, 4.0))
            if o.preserve_formant:
                filters.append(f"rubberband=pitch={ratio:.6f}:formant=preserved")
            else:
                filters.append(f"rubberband=pitch={ratio:.6f}")

        if abs(o.speed - 1.0) > 1e-6:
            speed = max(0.1, min(float(o.speed), 10.0))
            while speed > 2.0:
                filters.append("atempo=2.0")
                speed /= 2.0
            while speed < 0.5:
                filters.append("atempo=0.5")
                speed /= 0.5
            filters.append(f"atempo={speed:.6f}")

        if o.loudness_normalize:
            target = max(-70.0, min(float(o.loudness_target), -5.0))
            filters.append(f"loudnorm=I={target}:TP=-1.5:LRA=11")

        if abs(o.volume_db) > 1e-6:
            db = max(-40.0, min(float(o.volume_db), 20.0))
            filters.append(f"volume={db:.2f}dB")

        if o.trim_silence:
            filters.extend(
                [
                    "silenceremove=start_periods=1:start_duration=0:start_threshold=-50dB",
                    "areverse",
                    "silenceremove=start_periods=1:start_duration=0:start_threshold=-50dB",
                    "areverse",
                ]
            )

        if o.fade_ms > 0:
            d = max(1, min(int(o.fade_ms), 2000)) / 1000.0
            filters.append(f"afade=t=in:st=0:d={d:.3f}")
            filters.extend(["areverse", f"afade=t=in:st=0:d={d:.3f}", "areverse"])

        return ",".join(filters)

    async def process(
        self,
        audio: bytes,
        audio_format: str = "mp3",
        *,
        sample_rate: int = DEFAULT_PCM_SAMPLE_RATE,
    ) -> bytes:
        chain = self.build_filter_chain()
        if not chain or not audio:
            return audio

        ffmpeg = self._resolve_ffmpeg()
        if not ffmpeg:
            return audio

        fmt = (audio_format or "mp3").lower()
        rate = int(sample_rate) if sample_rate and int(sample_rate) > 0 else DEFAULT_PCM_SAMPLE_RATE

        if fmt in RAW_PCM_FORMATS:
            decoder_args = ["-f", "s16le", "-ar", str(rate), "-ac", "1"]
            encoder = ["-f", "s16le", "-c:a", "pcm_s16le", "-ar", str(rate), "-ac", "1"]
            suffix = ".pcm"
        else:
            encoder = _ENCODERS.get(fmt)
            if encoder is None:
                self.logger.warning("未知音频格式 %r，跳过音频处理", audio_format)
                return audio
            decoder_args = []
            suffix = f".{fmt}"

        tmpdir = None
        try:
            tmpdir = self._make_tmpdir()
            src = Path(tmpdir) / f"in{suffix}"
            dst = Path(tmpdir) / f"out{suffix}"
            src.write_bytes(audio)

            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
            cmd.extend(decoder_args)
            cmd.extend(["-i", str(src), "-af", chain])
            cmd.extend(encoder)
            cmd.append(str(dst))

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                self.logger.warning("ffmpeg 处理超时（>10s），本次跳过音频处理")
                return audio
            except asyncio.CancelledError:
                _kill_and_reap(proc)
                raise

            if proc.returncode != 0:
                msg = (stderr or b"").decode("utf-8", "ignore").strip()[:300]
                self.logger.warning("ffmpeg 处理失败（exit=%s），本次跳过音频处理: %s", proc.returncode, msg)
                return audio

            if not dst.is_file() or dst.stat().st_size == 0:
                self.logger.warning("ffmpeg 未产出有效音频，本次跳过音频处理")
                return audio

            result = dst.read_bytes()
            self.logger.info(
                "音频处理完成: %s -> %s 字节（滤镜: %s）",
                len(audio),
                len(result),
                chain[:120],
            )
            return result

        except FileNotFoundError:
            self.logger.warning("调用 ffmpeg 失败（文件不存在），本次跳过音频处理")
            return audio
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("音频处理异常，本次跳过: %r", exc)
            return audio
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)


def build_audio_processor(
    config: object,
    logger: Optional[logging.Logger] = None,
    work_dir: Optional[Path] = None,
) -> Optional[AudioProcessor]:
    section = getattr(config, "audio", None)
    if section is None:
        return None

    options = AudioOptions(
        ffmpeg_path=getattr(section, "ffmpeg_path", "ffmpeg") or "ffmpeg",
        pitch_semitones=float(getattr(section, "pitch_semitones", 0.0) or 0.0),
        preserve_formant=bool(getattr(section, "preserve_formant", True)),
        speed=float(getattr(section, "speed", 1.0) or 1.0),
        loudness_normalize=bool(getattr(section, "loudness_normalize", False)),
        loudness_target=float(getattr(section, "loudness_target", -16.0) or -16.0),
        highpass_hz=int(getattr(section, "highpass_hz", 0) or 0),
        volume_db=float(getattr(section, "volume_db", 0.0) or 0.0),
        fade_ms=int(getattr(section, "fade_ms", 0) or 0),
        trim_silence=bool(getattr(section, "trim_silence", False)),
    )
    if options.is_noop:
        return None
    return AudioProcessor(options, logger=logger, work_dir=work_dir)
