"""Fish Audio TTS MaiBot 插件离线单元测试套件。"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import logging
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import msgpack

PLUGIN_DIR = Path(__file__).resolve().parent.parent
PKG_NAME = "fishaudio_plugin_under_test"


def _load_plugin_package():
    if PKG_NAME in sys.modules:
        return sys.modules[PKG_NAME]

    init_spec = importlib.util.spec_from_file_location(
        PKG_NAME,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert init_spec is not None and init_spec.loader is not None
    pkg_mod = importlib.util.module_from_spec(init_spec)
    sys.modules[PKG_NAME] = pkg_mod
    init_spec.loader.exec_module(pkg_mod)

    for sub in ("audio_processor", "tts_service", "plugin"):
        full_name = f"{PKG_NAME}.{sub}"
        spec = importlib.util.spec_from_file_location(full_name, PLUGIN_DIR / f"{sub}.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)
    return pkg_mod


_load_plugin_package()
audio_processor_mod = sys.modules[f"{PKG_NAME}.audio_processor"]
tts_service_mod = sys.modules[f"{PKG_NAME}.tts_service"]
plugin_mod = sys.modules[f"{PKG_NAME}.plugin"]

AudioOptions = audio_processor_mod.AudioOptions
AudioProcessor = audio_processor_mod.AudioProcessor
build_audio_processor = audio_processor_mod.build_audio_processor
wrap_raw_pcm_to_wav = audio_processor_mod.wrap_raw_pcm_to_wav

FishAudioTTSService = tts_service_mod.FishAudioTTSService
FishSynthesisOptions = tts_service_mod.FishSynthesisOptions
ReferenceSample = tts_service_mod.ReferenceSample
resolve_tts_endpoint = tts_service_mod.resolve_tts_endpoint

FishAudioPluginConfig = plugin_mod.FishAudioPluginConfig
FishAudioVoicePlugin = plugin_mod.FishAudioVoicePlugin
apply_style_instruction = plugin_mod.apply_style_instruction
create_plugin = plugin_mod.create_plugin
split_tts_segments = plugin_mod.split_tts_segments
strip_audio_tags = plugin_mod.strip_audio_tags
_extract_outbound_plain_text = plugin_mod._extract_outbound_plain_text  # noqa: SLF001
_prune_output_dir_sync = plugin_mod._prune_output_dir_sync  # noqa: SLF001


class _FakeResponse:
    def __init__(
        self,
        status: int = 200,
        body: bytes = b"ID3dummy-mp3-bytes",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self.headers = headers or {"Content-Type": "audio/mpeg"}

    async def read(self) -> bytes:
        return self._body

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


def _make_dummy_ctx(streams: list[Any] | None = None) -> SimpleNamespace:
    send = SimpleNamespace(
        text=AsyncMock(return_value=SimpleNamespace(success=True, error="")),
        custom=AsyncMock(return_value=SimpleNamespace(success=True, error="")),
    )
    message = SimpleNamespace(get_by_id=AsyncMock(return_value=None))
    chat = SimpleNamespace(get_all_streams=AsyncMock(return_value=streams or []))
    paths = SimpleNamespace(data_dir=PLUGIN_DIR, runtime_dir=PLUGIN_DIR)
    logger = logging.getLogger("fishaudio.test")
    return SimpleNamespace(send=send, message=message, chat=chat, paths=paths, logger=logger)


class ConfigAndEndpointTests(unittest.TestCase):
    """配置模型与端点 URL 归一化测试。"""

    def test_default_config_matches_spec(self) -> None:
        cfg = FishAudioPluginConfig()
        self.assertTrue(cfg.plugin.enabled)
        self.assertEqual(cfg.voice.model, "s2.1-pro-free")
        self.assertEqual(cfg.voice.api_base_url, "https://api.fish.audio/v1/tts")
        self.assertEqual(cfg.voice.voice_mode, "auto")
        self.assertEqual(cfg.voice.reference_id, "")
        self.assertEqual(cfg.voice.audio_format, "mp3")
        self.assertEqual(cfg.generation.chunk_length, 300)
        self.assertEqual(cfg.generation.prosody_speed, 1.0)
        self.assertTrue(cfg.fallback.fallback_to_text)

    def test_example_toml_parses_against_schema(self) -> None:
        data = tomllib.loads((PLUGIN_DIR / "config.example.toml").read_text(encoding="utf-8"))
        cfg = FishAudioPluginConfig.model_validate(data)
        self.assertEqual(cfg.voice.fish_api_key, "")
        self.assertEqual(cfg.voice.model, "s2.1-pro-free")

    def test_resolve_tts_endpoint(self) -> None:
        self.assertEqual(resolve_tts_endpoint(""), "https://api.fish.audio/v1/tts")
        self.assertEqual(resolve_tts_endpoint("https://api.fish.audio"), "https://api.fish.audio/v1/tts")
        self.assertEqual(resolve_tts_endpoint("https://api.fish.audio/v1"), "https://api.fish.audio/v1/tts")
        self.assertEqual(resolve_tts_endpoint("https://api.fish.audio/v1/tts/"), "https://api.fish.audio/v1/tts")
        self.assertEqual(resolve_tts_endpoint("https://proxy.example.com/custom"), "https://proxy.example.com/custom/v1/tts")


class ServiceProtocolTests(unittest.IsolatedAsyncioTestCase):
    """HTTP 协议构造、MessagePack 编码与容错重试测试。"""

    def test_headers_and_json_payload(self) -> None:
        svc = FishAudioTTSService("sk-test-1234", "https://api.fish.audio")
        headers = svc.build_headers("s2.1-pro-free", use_msgpack=False)
        self.assertEqual(headers["Authorization"], "Bearer sk-test-1234")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["model"], "s2.1-pro-free")

        opts = FishSynthesisOptions(model="s2.1-pro-free", audio_format="mp3", mp3_bitrate=192)
        payload = svc.build_payload("你好世界", opts, reference_id="ref_a, ref_b")
        self.assertEqual(payload["text"], "你好世界")
        self.assertEqual(payload["format"], "mp3")
        self.assertEqual(payload["mp3_bitrate"], 192)
        self.assertNotIn("opus_bitrate", payload)
        self.assertEqual(payload["reference_id"], ["ref_a", "ref_b"])
        self.assertEqual(payload["prosody"]["speed"], 1.0)

    def test_msgpack_reference_encoding(self) -> None:
        svc = FishAudioTTSService("sk-test-1234")
        headers = svc.build_headers("s2.1-pro", use_msgpack=True)
        self.assertEqual(headers["Content-Type"], "application/msgpack")

        opts = FishSynthesisOptions(model="s2.1-pro", audio_format="wav", sample_rate=24000)
        payload = svc.build_payload(
            "克隆测试",
            opts,
            reference_id="ignored_when_refs_present",
            references=[ReferenceSample(audio=b"RIFFdata", text="原始台词")],
        )
        self.assertNotIn("reference_id", payload)
        self.assertNotIn("mp3_bitrate", payload)
        self.assertEqual(payload["sample_rate"], 24000)

        packed = svc.encode_body(payload, use_msgpack=True)
        unpacked = msgpack.unpackb(packed, raw=False)
        self.assertEqual(unpacked["references"][0]["audio"], b"RIFFdata")
        self.assertEqual(unpacked["references"][0]["text"], "原始台词")

    async def test_retry_on_429_then_success(self) -> None:
        svc = FishAudioTTSService("sk-test-1234", max_retries=2, retry_backoff=0.01)
        responses = [
            _FakeResponse(status=429, body=b'{"message": "rate limit"}', headers={"Content-Type": "application/json"}),
            _FakeResponse(status=200, body=b"FAKE_MP3_OK", headers={"Content-Type": "audio/mpeg"}),
        ]
        call_count = 0

        def _post(*args: Any, **kwargs: Any) -> _FakeResponse:
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return resp

        fake_session = MagicMock()
        fake_session.closed = False
        fake_session.post = MagicMock(side_effect=_post)
        fake_session.close = AsyncMock()
        svc._session = fake_session  # noqa: SLF001

        res = await svc.synthesize("测试重试")
        self.assertTrue(res["success"])
        self.assertEqual(call_count, 2)
        self.assertEqual(base64.b64decode(res["audio_base64"]), b"FAKE_MP3_OK")
        await svc.close()

    async def test_no_retry_on_401_or_timeout_or_json_200(self) -> None:
        svc = FishAudioTTSService("sk-test-1234", max_retries=2, retry_backoff=0.01)
        fake_session = MagicMock()
        fake_session.closed = False
        fake_session.close = AsyncMock()
        svc._session = fake_session  # noqa: SLF001

        # 401 不应重试
        fake_session.post = MagicMock(
            return_value=_FakeResponse(
                status=401,
                body=b'{"message": "invalid api key"}',
                headers={"Content-Type": "application/json"},
            )
        )
        res_401 = await svc.synthesize("鉴权失败")
        self.assertFalse(res_401["success"])
        self.assertEqual(fake_session.post.call_count, 1)

        # HTTP 200 但返回 JSON 错误应拦截
        fake_session.post = MagicMock(
            return_value=_FakeResponse(
                status=200,
                body=b'{"message": "insufficient credits"}',
                headers={"Content-Type": "application/json"},
            )
        )
        res_json = await svc.synthesize("异常返回")
        self.assertFalse(res_json["success"])
        self.assertIn("insufficient credits", res_json["error"])

        # 超时不重试
        fake_session.post = MagicMock(side_effect=asyncio.TimeoutError())
        res_to = await svc.synthesize("超时测试")
        self.assertFalse(res_to["success"])
        self.assertEqual(fake_session.post.call_count, 1)
        await svc.close()


class AudioAndTagHelpersTests(unittest.IsolatedAsyncioTestCase):
    """音频处理与情感标签辅助函数测试。"""

    def test_wrap_raw_pcm_to_wav_headers(self) -> None:
        pcm = b"\x00\x01" * 100
        wav_44k = wrap_raw_pcm_to_wav(pcm)
        self.assertEqual(wav_44k[:4], b"RIFF")
        self.assertEqual(wav_44k[8:12], b"WAVE")
        self.assertEqual(int.from_bytes(wav_44k[24:28], "little"), 44100)
        self.assertEqual(len(wav_44k), 44 + len(pcm))

        wav_24k = wrap_raw_pcm_to_wav(pcm, sample_rate=24000)
        self.assertEqual(int.from_bytes(wav_24k[24:28], "little"), 24000)

    def test_audio_filter_chain_and_noop(self) -> None:
        cfg = FishAudioPluginConfig()
        self.assertIsNone(build_audio_processor(cfg))

        proc = AudioProcessor(
            AudioOptions(
                ffmpeg_path="ffmpeg",
                pitch_semitones=2.0,
                preserve_formant=True,
                speed=1.1,
                loudness_normalize=True,
                highpass_hz=80,
                volume_db=2.5,
                fade_ms=50,
                trim_silence=True,
            )
        )
        chain = proc.build_filter_chain()
        self.assertIn("highpass=f=80", chain)
        self.assertIn("rubberband=", chain)
        self.assertIn("loudnorm=", chain)
        self.assertIn("afade=", chain)

    async def test_audio_processor_kills_subprocess_on_cancel(self) -> None:
        proc_opts = AudioOptions(ffmpeg_path="ffmpeg", highpass_hz=80)
        processor = AudioProcessor(proc_opts)

        fake_child = MagicMock()
        fake_child.returncode = None
        fake_child.kill = MagicMock()
        fake_child.wait = AsyncMock(return_value=-9)

        async def slow_comm() -> tuple[bytes, bytes]:
            await asyncio.sleep(5)
            return b"", b""

        fake_child.communicate = slow_comm

        with patch.object(processor, "_resolve_ffmpeg", return_value="ffmpeg"), patch(
            "asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake_child)
        ):
            task = asyncio.create_task(processor.process(b"dummy-mp3", "mp3"))
            await asyncio.sleep(0.02)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            fake_child.kill.assert_called_once()

    def test_style_and_tag_helpers(self) -> None:
        self.assertEqual(apply_style_instruction("你好", "", "s2.1-pro-free"), "你好")
        self.assertEqual(apply_style_instruction("你好", "whispering", "s2.1-pro-free"), "[whispering] 你好")
        self.assertEqual(apply_style_instruction("你好", "[gentle]", "s2.1-pro"), "[gentle] 你好")
        self.assertEqual(apply_style_instruction("你好", "whispering", "s1"), "(whispering) 你好")

        cleaned = strip_audio_tags("[laughing] 早安呀 （轻笑） <style>温柔</style> 第(1)句")
        self.assertEqual(cleaned, "早安呀 第(1)句")


class PluginOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    """插件音色解析、防串群流定位、降级发文字与热重载测试。"""

    def _build_plugin(self, streams: list[Any] | None = None, **voice_overrides: Any) -> FishAudioVoicePlugin:
        plugin = create_plugin()
        cfg = FishAudioPluginConfig()
        cfg.voice.fish_api_key = "sk-test-token"
        for k, v in voice_overrides.items():
            setattr(cfg.voice, k, v)
        plugin._plugin_config_instance = cfg
        plugin._set_context(_make_dummy_ctx(streams=streams))
        plugin._rebuild_runtime()  # noqa: SLF001
        return plugin

    def test_voice_resolution_with_temp_voices_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vdir = Path(tmp)
            (vdir / "alice.wav").write_bytes(b"RIFF_ALICE_WAV")
            (vdir / "alice.txt").write_text("我是爱丽丝参考音频。", encoding="utf-8")

            # auto + reference_id 为空 -> 自动选中本地 clone
            p1 = self._build_plugin(voices_dir=str(vdir), voice_mode="auto", reference_id="")
            label, rid, refs = p1._resolve_voice_target("")  # noqa: SLF001
            self.assertEqual(label, "clone:alice")
            self.assertEqual(rid, "")
            assert refs is not None
            self.assertEqual(refs[0].audio, b"RIFF_ALICE_WAV")
            self.assertEqual(refs[0].text, "我是爱丽丝参考音频。")

            # auto + reference_id 非空 -> 优先云端 reference_id
            p2 = self._build_plugin(voices_dir=str(vdir), voice_mode="auto", reference_id="fish_id_1")
            label2, rid2, refs2 = p2._resolve_voice_target("")  # noqa: SLF001
            self.assertEqual(label2, "reference_id:fish_id_1")
            self.assertEqual(rid2, "fish_id_1")
            self.assertIsNone(refs2)

            # 尝试路径穿越字符串不会读本地任意文件，只作字符串处理
            label3, rid3, refs3 = p1._resolve_voice_target("../secret")  # noqa: SLF001
            self.assertIsNone(refs3)
            self.assertEqual(rid3, "../secret")

    async def test_find_stream_id_anti_cross_chat(self) -> None:
        # 多个聊天流且无 stream_id/message_id 时必须拒绝猜测并返回空串
        p_multi = self._build_plugin(streams=[{"stream_id": "s1"}, {"stream_id": "s2"}])
        self.assertEqual(await p_multi._find_stream_id({}), "")  # noqa: SLF001

        # 唯一聊天流允许回退定位
        p_single = self._build_plugin(streams=[{"stream_id": "only_stream"}])
        self.assertEqual(await p_single._find_stream_id({}), "only_stream")  # noqa: SLF001

        # message_id 通过 ctx.message.get_by_id 查询
        p_multi.ctx.message.get_by_id = AsyncMock(return_value={"chat_id": "from_msg_42"})
        self.assertEqual(
            await p_multi._find_stream_id({"message_id": "m1"}),  # noqa: SLF001
            "from_msg_42",
        )

    async def test_send_voice_reply_success_and_pcm_wrapping(self) -> None:
        plugin = self._build_plugin(audio_format="pcm", sample_rate=24000)
        pcm_raw = b"\x01\x00" * 50
        plugin._tts_service.synthesize = AsyncMock(  # noqa: SLF001
            return_value={
                "success": True,
                "audio_base64": base64.b64encode(pcm_raw).decode("ascii"),
                "audio_bytes_len": len(pcm_raw),
                "text": "你好",
            }
        )

        res = await plugin.send_voice_reply(text="你好", style="gentle", stream_id="group_1001")
        self.assertTrue(res["success"])
        self.assertEqual(res["format"], "wav")
        plugin.ctx.send.custom.assert_awaited_once()
        sent_type, sent_b64, sent_stream = plugin.ctx.send.custom.await_args.args
        self.assertEqual(sent_type, "voice")
        self.assertEqual(sent_stream, "group_1001")
        decoded = base64.b64decode(sent_b64)
        self.assertEqual(decoded[:4], b"RIFF")
        self.assertEqual(int.from_bytes(decoded[24:28], "little"), 24000)

    async def test_send_voice_reply_fallback_strips_tags_and_uses_placeholder(self) -> None:
        plugin = self._build_plugin()
        plugin._tts_service.synthesize = AsyncMock(  # noqa: SLF001
            return_value={"success": False, "error": "upstream 503"}
        )

        # 正常台词带风格参数时，降级发送纯正原文并剥离方括号标签
        res = await plugin.send_voice_reply(
            text="[laughing] 今天天气真好",
            style="happy",
            stream_id="private_99",
        )
        self.assertFalse(res["success"])
        self.assertTrue(res["fallback_sent"])
        plugin.ctx.send.text.assert_awaited_with("今天天气真好", "private_99")

        # 纯标签被剥光时启用保底文案
        res2 = await plugin.send_voice_reply(
            text="[sighs] （叹气）",
            stream_id="private_99",
        )
        self.assertTrue(res2["fallback_sent"])
        plugin.ctx.send.text.assert_awaited_with("（想说点什么，但声音没发出来）", "private_99")

    async def test_busy_skip_and_runtime_disable_hot_reload(self) -> None:
        plugin = self._build_plugin()
        await plugin._semaphore.acquire()  # noqa: SLF001
        res_busy = await plugin.send_voice_reply(text="排队时跳过", stream_id="g1")
        self.assertFalse(res_busy["success"])
        self.assertEqual(res_busy["status"], "fallback_text")
        plugin.ctx.send.text.assert_awaited_with("排队时跳过", "g1")
        plugin._semaphore.release()  # noqa: SLF001

        # 热重载关闭插件后不再合成也不发声
        old_service = plugin._tts_service  # noqa: SLF001
        old_service.close = AsyncMock()
        plugin.config.plugin.enabled = False
        await plugin.on_config_update(scope="self")
        old_service.close.assert_awaited_once()

        res_disabled = await plugin.send_voice_reply(text="禁用测试", stream_id="g1")
        self.assertEqual(res_disabled["status"], "disabled")


class AutoVoiceAndStorageTests(unittest.IsolatedAsyncioTestCase):
    """默认文字转语音、智能分段与存储自动清理测试。"""

    def _build_plugin(self, **overrides: Any) -> FishAudioVoicePlugin:
        plugin = create_plugin()
        cfg = FishAudioPluginConfig()
        cfg.voice.fish_api_key = "sk-test-token"
        for section_name, fields in overrides.items():
            section = getattr(cfg, section_name)
            for k, v in fields.items():
                setattr(section, k, v)
        plugin._plugin_config_instance = cfg
        plugin._set_context(_make_dummy_ctx(streams=[]))
        plugin._rebuild_runtime()  # noqa: SLF001
        return plugin

    def test_split_tts_segments_sentences_quotes_and_limits(self) -> None:
        # 多句中文含引号拆分，不在数字小数点处误切
        text = "“今天pi大约是3.14！”小鱼笑着说。明天我们去看电影好不好？好的"
        segs = split_tts_segments(text, max_chars=80, split_sentences=True)
        self.assertEqual(
            segs,
            [
                "“今天pi大约是3.14！”小鱼笑着说。",
                "明天我们去看电影好不好？",
                "好的",
            ],
        )

        # 超长无句号按逗号/最大字数兜底拆段
        long_clause = "第一段很长的内容，第二段也很长，第三段继续延续，第四段结尾收束。"
        packed = split_tts_segments(long_clause, max_chars=18, split_sentences=True)
        self.assertTrue(all(len(s) <= 18 for s in packed))
        self.assertGreaterEqual(len(packed), 2)

        # 关闭分段时保持整段
        self.assertEqual(
            split_tts_segments("你好。世界。", max_chars=20, split_sentences=False),
            ["你好。世界。"],
        )
        # 纯标点跳过
        self.assertEqual(split_tts_segments("...！？", max_chars=80, split_sentences=True), [])

    def test_extract_outbound_plain_text_filters_media_and_at(self) -> None:
        self.assertEqual(
            _extract_outbound_plain_text(
                {
                    "raw_message": [
                        {"type": "at", "data": {"target_user_nickname": "小明"}},
                        {"type": "reply", "data": {"target_message_id": "m0"}},
                        {"type": "text", "data": "@小明 早上好呀！"},
                    ]
                }
            ),
            "早上好呀！",
        )
        # 已是语音或包含图片/文件时不得二次转换
        self.assertIsNone(
            _extract_outbound_plain_text({"raw_message": [{"type": "voice", "binary_data_base64": "abc"}]})
        )
        self.assertIsNone(
            _extract_outbound_plain_text(
                {
                    "raw_message": [
                        {"type": "text", "data": "看图"},
                        {"type": "image", "data": "img"},
                    ]
                }
            )
        )

    async def test_auto_voice_hook_single_and_multi_segment(self) -> None:
        plugin = self._build_plugin(auto_voice={"enabled": True, "segment_delay_ms": 0, "max_segment_chars": 80})

        async def fake_synth(text: str, **kwargs: Any) -> dict[str, Any]:
            raw = f"AUDIO:{text}".encode("utf-8")
            return {
                "success": True,
                "audio_base64": base64.b64encode(raw).decode("ascii"),
                "audio_bytes_len": len(raw),
                "text": text,
            }

        plugin._tts_service.synthesize = AsyncMock(side_effect=fake_synth)  # noqa: SLF001

        msg = {
            "session_id": "group_88",
            "raw_message": [{"type": "text", "data": "第一句你好。第二句再见！"}],
            "processed_plain_text": "第一句你好。第二句再见！",
        }
        res = await plugin.auto_voice_hook(
            message=msg,
            typing=False,
            set_reply=True,
            reply_message_id="orig_1",
            storage_message=True,
            show_log=True,
        )
        assert res is not None
        self.assertEqual(res["action"], "continue")
        # 前 1 段通过 send.custom 发出，第 2 段通过 modified_kwargs 原位发出并关闭 set_reply
        plugin.ctx.send.custom.assert_awaited_once()
        c_type, c_b64, c_stream = plugin.ctx.send.custom.await_args.args
        self.assertEqual(c_type, "voice")
        self.assertEqual(c_stream, "group_88")
        self.assertEqual(base64.b64decode(c_b64), "AUDIO:第一句你好。".encode("utf-8"))

        mod_kwargs = res["modified_kwargs"]
        self.assertFalse(mod_kwargs["set_reply"])
        self.assertEqual(mod_kwargs["reply_message_id"], "")
        mod_msg = mod_kwargs["message"]
        self.assertEqual(mod_msg["processed_plain_text"], "第二句再见！")
        self.assertEqual(mod_msg["raw_message"][0]["type"], "voice")
        self.assertEqual(mod_msg["raw_message"][0]["data"], "第二句再见！")
        self.assertEqual(
            base64.b64decode(mod_msg["raw_message"][0]["binary_data_base64"]),
            "AUDIO:第二句再见！".encode("utf-8"),
        )

    async def test_auto_voice_hook_skip_url_partial_failure_and_fallback_bypass(self) -> None:
        plugin = self._build_plugin(auto_voice={"enabled": True, "segment_delay_ms": 0})
        plugin._tts_service.synthesize = AsyncMock(  # noqa: SLF001
            side_effect=[
                {
                    "success": True,
                    "audio_base64": base64.b64encode(b"SEG1").decode("ascii"),
                    "audio_bytes_len": 4,
                    "text": "前一句。",
                },
                {"success": False, "error": "upstream 503"},
            ]
        )

        # 含 URL 直接透传不转语音
        url_msg = {"session_id": "s1", "raw_message": [{"type": "text", "data": "点击 https://example.com"}]}
        self.assertIsNone(await plugin.auto_voice_hook(message=url_msg, set_reply=False))

        # 降级文本防环回令牌放行
        plugin._register_fallback_bypass("s1", "保底纯文本")  # noqa: SLF001
        fb_msg = {"session_id": "s1", "raw_message": [{"type": "text", "data": "保底纯文本"}]}
        self.assertIsNone(await plugin.auto_voice_hook(message=fb_msg, set_reply=False))

        # 多段中第 2 段合成失败：第 1 段已转语音发出，第 2 段以纯文本原样保留放行
        partial_msg = {
            "session_id": "s1",
            "raw_message": [{"type": "text", "data": "前一句。后一句失败"}],
        }
        partial_res = await plugin.auto_voice_hook(message=partial_msg, set_reply=False)
        assert partial_res is not None
        plugin.ctx.send.custom.assert_awaited_once()
        out_raw = partial_res["modified_kwargs"]["message"]["raw_message"]
        self.assertEqual(out_raw, [{"type": "text", "data": "后一句失败"}])

    async def test_storage_save_immediate_delete_and_safe_prune(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            # 非插件自有文件绝不能被误删
            user_file = out_dir / "keep_me.mp3"
            user_file.write_bytes(b"USER_OWNED")

            # 保存 + retain_minutes=0 立即自动删除
            p_imm = self._build_plugin(
                storage={
                    "save_output_audio": True,
                    "output_dir": str(out_dir),
                    "auto_delete": True,
                    "retain_minutes": 0,
                    "max_saved_files": 100,
                }
            )
            p_imm._tts_service.synthesize = AsyncMock(  # noqa: SLF001
                return_value={
                    "success": True,
                    "audio_base64": base64.b64encode(b"FAKE_MP3").decode("ascii"),
                    "audio_bytes_len": 8,
                    "text": "测试存储",
                }
            )
            res = await p_imm.api_synthesize("测试存储")
            self.assertTrue(res["success"])
            self.assertNotIn("audio_path", res)
            self.assertTrue(user_file.exists())
            leftover = [p for p in out_dir.iterdir() if p.name.startswith("fishaudio_")]
            self.assertEqual(leftover, [])

            # 保存 + max_saved_files 数量上限淘汰最旧文件
            old_a = out_dir / "fishaudio_20261111_010000_aaaaaaaa.mp3"
            old_b = out_dir / "fishaudio_20261111_020000_bbbbbbbb.mp3"
            new_c = out_dir / "fishaudio_20261111_030000_cccccccc.mp3"
            old_a.write_bytes(b"a")
            old_b.write_bytes(b"b")
            new_c.write_bytes(b"c")
            import os, time as _t

            now = _t.time()
            os.utime(old_a, (now - 300, now - 300))
            os.utime(old_b, (now - 200, now - 200))
            os.utime(new_c, (now - 100, now - 100))

            removed = _prune_output_dir_sync(out_dir, retain_minutes=60, max_saved_files=2)
            self.assertEqual(removed, 1)
            self.assertFalse(old_a.exists())
            self.assertTrue(old_b.exists())
            self.assertTrue(new_c.exists())
            self.assertTrue(user_file.exists())

    async def test_extract_dict_text_and_partial_send_no_drop(self) -> None:
        # 兼容字典形态 text 段：{"type": "text", "data": {"text": "..."}}
        self.assertEqual(
            _extract_outbound_plain_text(
                {"raw_message": [{"type": "text", "data": {"text": "适配器字典文本"}}]}
            ),
            "适配器字典文本",
        )

        # 三段消息：第3段合成失败，且前2段发送时第2段发送中断 -> 必须保留第2段与第3段文本
        plugin = self._build_plugin(auto_voice={"enabled": True, "segment_delay_ms": 0})
        plugin._tts_service.synthesize = AsyncMock(  # noqa: SLF001
            side_effect=[
                {"success": True, "audio_base64": base64.b64encode(b"S1").decode("ascii"), "audio_bytes_len": 2, "text": "第一句。"},
                {"success": True, "audio_base64": base64.b64encode(b"S2").decode("ascii"), "audio_bytes_len": 2, "text": "第二句。"},
                {"success": False, "error": "upstream down"},
            ]
        )
        plugin.ctx.send.custom = AsyncMock(
            side_effect=[
                SimpleNamespace(success=True, error=""),
                SimpleNamespace(success=False, error="send drop"),
            ]
        )
        hook_msg = {"session_id": "s1", "raw_message": [{"type": "text", "data": "第一句。第二句。第三句。"}]}
        res = await plugin.auto_voice_hook(message=hook_msg, set_reply=False)
        assert res is not None
        out_raw = res["modified_kwargs"]["message"]["raw_message"]
        self.assertEqual(out_raw, [{"type": "text", "data": "第二句。\n第三句。"}])

        # send_voice_reply 同场景也不能丢中间段（文本长度需超过 max_segment_chars 才拆段）
        plugin_tool = self._build_plugin(auto_voice={"enabled": True, "max_segment_chars": 20, "segment_delay_ms": 0})
        plugin_tool._tts_service.synthesize = AsyncMock(  # noqa: SLF001
            side_effect=[
                {"success": True, "audio_base64": base64.b64encode(b"S1").decode("ascii"), "audio_bytes_len": 2, "text": "第一句较长凑够字数。"},
                {"success": True, "audio_base64": base64.b64encode(b"S2").decode("ascii"), "audio_bytes_len": 2, "text": "第二句很长凑够字数。"},
                {"success": False, "error": "upstream down"},
            ]
        )
        plugin_tool.ctx.send.custom = AsyncMock(
            side_effect=[
                SimpleNamespace(success=True, error=""),
                SimpleNamespace(success=False, error="send drop"),
            ]
        )
        tool_res = await plugin_tool.send_voice_reply(
            text="第一句较长凑够字数。第二句很长凑够字数。第三句。",
            stream_id="s1",
        )
        self.assertFalse(tool_res["success"])
        self.assertEqual(tool_res["partial_voice_segments"], 1)
        sent_fb = plugin_tool.ctx.send.text.await_args[0][0]
        self.assertIn("第二句很长凑够字数。", sent_fb)
        self.assertIn("第三句。", sent_fb)

    async def test_hook_unexpected_error_no_duplicate_send_and_cleans_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            plugin = self._build_plugin(
                auto_voice={"enabled": True, "segment_delay_ms": 10},
                storage={
                    "save_output_audio": True,
                    "output_dir": str(out_dir),
                    "auto_delete": True,
                    "retain_minutes": 0,
                },
            )
            plugin._tts_service.synthesize = AsyncMock(  # noqa: SLF001
                side_effect=[
                    {"success": True, "audio_base64": base64.b64encode(b"S1").decode("ascii"), "audio_bytes_len": 2, "text": "一句。"},
                    {"success": True, "audio_base64": base64.b64encode(b"S2").decode("ascii"), "audio_bytes_len": 2, "text": "二句。"},
                    {"success": True, "audio_base64": base64.b64encode(b"S3").decode("ascii"), "audio_bytes_len": 2, "text": "三句。"},
                ]
            )
            # 先真实发出第 1 段，在发第 2 段前 asyncio.sleep 抛异常 -> 第 2、3 段转为文本发出且 3 个临时文件均清理
            with unittest.mock.patch(
                "asyncio.sleep", new=AsyncMock(side_effect=RuntimeError("unexpected loop error"))
            ):
                res = await plugin.auto_voice_hook(
                    message={"session_id": "s1", "raw_message": [{"type": "text", "data": "一句。二句。三句。"}]},
                    set_reply=False,
                )
            assert res is not None
            plugin.ctx.send.custom.assert_awaited_once()
            self.assertEqual(
                res["modified_kwargs"]["message"]["raw_message"],
                [{"type": "text", "data": "二句。\n三句。"}],
            )
            leftover = [p for p in out_dir.iterdir() if p.name.startswith("fishaudio_")]
            self.assertEqual(leftover, [])

            # 验证分段总耗时预算保护会在超时前及时停止后续合成，避免触及 90s 宿主强杀
            async def _slow_synth(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
                await asyncio.sleep(0.1)
                return {"success": True, "audio_base64": base64.b64encode(b"S").decode("ascii"), "audio_bytes_len": 1, "text": "x"}

            plugin._tts_service.synthesize = _slow_synth  # noqa: SLF001
            synthesized, failed_idx, err_msg = await plugin._synthesize_segments(  # noqa: SLF001
                ["第一段", "第二段"],
                total_budget_sec=0.02,
            )
            self.assertEqual(synthesized, [])
            self.assertEqual(failed_idx, 0)
            self.assertIn("预算", err_msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
