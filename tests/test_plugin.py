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
format_emotion_tag = plugin_mod.format_emotion_tag
has_inline_emotion_tags = plugin_mod.has_inline_emotion_tags
inject_emotion_tags_by_rules = plugin_mod.inject_emotion_tags_by_rules
map_annotated_to_segments = plugin_mod.map_annotated_to_segments
reconcile_injected_text = plugin_mod.reconcile_injected_text
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


def _make_dummy_ctx(streams: list[Any] | None = None, llm: Any = None) -> SimpleNamespace:
    send = SimpleNamespace(
        text=AsyncMock(return_value=SimpleNamespace(success=True, error="")),
        custom=AsyncMock(return_value=SimpleNamespace(success=True, error="")),
    )
    message = SimpleNamespace(get_by_id=AsyncMock(return_value=None))
    chat = SimpleNamespace(get_all_streams=AsyncMock(return_value=streams or []))
    llm_cap = llm or SimpleNamespace(generate=AsyncMock(return_value={"success": False, "response": ""}))
    paths = SimpleNamespace(data_dir=PLUGIN_DIR, runtime_dir=PLUGIN_DIR)
    logger = logging.getLogger("fishaudio.test")
    return SimpleNamespace(send=send, message=message, chat=chat, llm=llm_cap, paths=paths, logger=logger)


class ConfigAndEndpointTests(unittest.TestCase):
    """配置模型与端点 URL 归一化测试。"""

    def test_default_config_matches_spec(self) -> None:
        cfg = FishAudioPluginConfig()
        self.assertTrue(cfg.plugin.enabled)
        self.assertEqual(cfg.plugin.config_version, "1.2.0")
        self.assertEqual(cfg.voice.model, "s2.1-pro-free")
        self.assertEqual(cfg.voice.api_base_url, "https://api.fish.audio/v1/tts")
        self.assertEqual(cfg.voice.voice_mode, "auto")
        self.assertEqual(cfg.voice.reference_id, "")
        self.assertEqual(cfg.voice.audio_format, "mp3")
        self.assertEqual(cfg.auto_voice.mode, "planner")
        self.assertAlmostEqual(cfg.auto_voice.planner_temperature, 0.5)
        self.assertAlmostEqual(cfg.auto_voice.random_probability, 0.5)
        self.assertAlmostEqual(cfg.auto_voice.decision_ttl_seconds, 8.0)
        self.assertTrue(cfg.emotion.enabled)
        self.assertEqual(cfg.emotion.mode, "hybrid")
        self.assertTrue(cfg.emotion.skip_if_already_tagged)
        self.assertTrue(cfg.emotion.show_injected_text)
        self.assertFalse(cfg.emotion.send_injected_text_to_chat)
        self.assertEqual(cfg.generation.chunk_length, 300)
        self.assertEqual(cfg.generation.prosody_speed, 1.0)
        self.assertTrue(cfg.fallback.fallback_to_text)

    def test_example_and_active_toml_parse_against_schema(self) -> None:
        data = tomllib.loads((PLUGIN_DIR / "config.example.toml").read_text(encoding="utf-8"))
        cfg = FishAudioPluginConfig.model_validate(data)
        self.assertEqual(cfg.voice.fish_api_key, "")
        self.assertEqual(cfg.voice.model, "s2.1-pro-free")
        self.assertEqual(cfg.auto_voice.mode, "planner")
        self.assertAlmostEqual(cfg.auto_voice.planner_temperature, 0.5)
        self.assertAlmostEqual(cfg.auto_voice.random_probability, 0.5)

        active_path = PLUGIN_DIR / "config.toml"
        if active_path.is_file():
            active_data = tomllib.loads(active_path.read_text(encoding="utf-8"))
            active_cfg = FishAudioPluginConfig.model_validate(active_data)
            self.assertEqual(active_cfg.auto_voice.mode, "planner")

    def test_webui_schema_sliders_and_legacy_config_migration(self) -> None:
        plugin = create_plugin()
        schema = plugin.get_webui_config_schema(plugin_id="custom.maibot-fishaudio-voice-pro")
        av_fields = schema["sections"]["auto_voice"]["fields"]

        self.assertEqual(av_fields["mode"]["type"], "select")
        self.assertEqual(av_fields["mode"]["choices"], ["planner", "probability", "always", "off"])
        self.assertIn("choice:planner", av_fields["mode"]["i18n"]["zh-CN"])

        for slider_name in ("planner_temperature", "random_probability"):
            field_schema = av_fields[slider_name]
            self.assertEqual(field_schema["ui_type"], "slider")
            self.assertEqual(field_schema["min"], 0.0)
            self.assertEqual(field_schema["max"], 1.0)
            self.assertEqual(field_schema["step"], 0.05)

        emo_fields = schema["sections"]["emotion"]["fields"]
        self.assertTrue(emo_fields["enabled"]["default"])
        self.assertEqual(emo_fields["mode"]["choices"], ["hybrid", "llm", "rule"])
        self.assertEqual(emo_fields["temperature"]["ui_type"], "slider")

        # 旧版 1.0.0 enabled=true/false 迁移
        migrated_on, changed_on = plugin.normalize_plugin_config(
            {"plugin": {"config_version": "1.0.0", "enabled": True}, "auto_voice": {"enabled": True}}
        )
        self.assertTrue(changed_on)
        self.assertEqual(migrated_on["auto_voice"]["mode"], "always")
        self.assertNotIn("enabled", migrated_on["auto_voice"])

        migrated_off, changed_off = plugin.normalize_plugin_config(
            {"plugin": {"config_version": "1.0.0", "enabled": True}, "auto_voice": {"enabled": False}}
        )
        self.assertTrue(changed_off)
        self.assertEqual(migrated_off["auto_voice"]["mode"], "off")

        # 显式校验组件收集（ReplyExtension、Command 与可见 Tool）
        components = {c["name"]: c for c in plugin.get_components()}
        self.assertIn("fishaudio", components)
        self.assertEqual(components["fishaudio"]["type"], "COMMAND")
        self.assertIn("voice_reply", components)
        self.assertEqual(components["voice_reply"]["type"], "REPLY_EXTENSION")
        self.assertEqual(components["fish_planner_voice_policy"]["type"], "HOOK_HANDLER")
        self.assertEqual(components["send_voice_reply"]["metadata"].get("visibility"), "visible")

    def test_resolve_tts_endpoint(self) -> None:
        self.assertEqual(resolve_tts_endpoint(""), "https://api.fish.audio/v1/tts")
        self.assertEqual(resolve_tts_endpoint("https://api.fish.audio"), "https://api.fish.audio/v1/tts")
        self.assertEqual(resolve_tts_endpoint("https://api.fish.audio/v1"), "https://api.fish.audio/v1/tts")
        self.assertEqual(resolve_tts_endpoint("https://api.fish.audio/v1/tts/"), "https://api.fish.audio/v1/tts")
        self.assertEqual(resolve_tts_endpoint("https://proxy.example.com/custom"), "https://proxy.example.com/custom/v1/tts")

    def test_pydantic_legacy_config_migration(self) -> None:
        """测试 AutoVoiceSectionConfig 在 Pydantic 校验阶段对旧版 enabled 开关的平滑迁移。"""
        cfg_enabled = plugin_mod.AutoVoiceSectionConfig.model_validate({"enabled": True})
        self.assertEqual(cfg_enabled.mode, "always")

        cfg_disabled = plugin_mod.AutoVoiceSectionConfig.model_validate({"enabled": False})
        self.assertEqual(cfg_disabled.mode, "off")

        top_enabled = FishAudioPluginConfig.model_validate({"auto_voice": {"enabled": True}})
        self.assertEqual(top_enabled.auto_voice.mode, "always")

        top_disabled = FishAudioPluginConfig.model_validate({"auto_voice": {"enabled": False}})
        self.assertEqual(top_disabled.auto_voice.mode, "off")


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

    async def test_retry_resets_session_on_connect_timeout_and_client_error(self) -> None:
        svc = FishAudioTTSService("sk-test-1234", max_retries=2, retry_backoff=0.01)

        s1 = MagicMock()
        s1.closed = False
        s1.close = AsyncMock()
        conn_key = aiohttp.client_reqrep.ConnectionKey("api.fish.audio", 443, True, True, None, None, None)
        opaque_err = aiohttp.ClientConnectorError(conn_key, OSError(10060, None))
        s1.post = MagicMock(side_effect=opaque_err)

        s2 = MagicMock()
        s2.closed = False
        s2.close = AsyncMock()
        s2.post = MagicMock(side_effect=aiohttp.ConnectionTimeoutError("connect stalled"))

        s3 = MagicMock()
        s3.closed = False
        s3.close = AsyncMock()
        s3.post = MagicMock(return_value=_FakeResponse(status=200, body=b"RECOVERED_MP3"))

        sessions = [s1, s2, s3]
        idx = 0

        async def fake_get_session():
            nonlocal idx
            if svc._session is None or svc._session.closed:  # noqa: SLF001
                svc._session = sessions[idx]  # noqa: SLF001
                idx += 1
            return svc._session  # noqa: SLF001

        with patch.object(svc, "_get_session", side_effect=fake_get_session):
            res = await svc.synthesize("网络恢复测试")

        self.assertTrue(res["success"])
        self.assertEqual(idx, 3)
        s1.close.assert_awaited_once()
        s2.close.assert_awaited_once()
        self.assertEqual(base64.b64decode(res["audio_base64"]), b"RECOVERED_MP3")
        formatted = svc._format_client_error(opaque_err)  # noqa: SLF001
        self.assertIn("errno=10060", formatted)
        await svc.close()

        # 验证真实 _get_session 启用 trust_env=True
        real_svc = FishAudioTTSService("sk-test-1234")
        real_sess = await real_svc._get_session()  # noqa: SLF001
        self.assertTrue(real_sess.trust_env)
        await real_svc.close()


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
        plugin = self._build_plugin(
            auto_voice={"mode": "always", "segment_delay_ms": 0, "max_segment_chars": 80},
            emotion={"enabled": False},
        )

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
        plugin = self._build_plugin(auto_voice={"mode": "always", "segment_delay_ms": 0})
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
        plugin = self._build_plugin(auto_voice={"mode": "always", "segment_delay_ms": 0})
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
        plugin_tool = self._build_plugin(auto_voice={"mode": "always", "max_segment_chars": 20, "segment_delay_ms": 0})
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
                auto_voice={"mode": "always", "segment_delay_ms": 10},
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


class PlannerAndProbabilityModeTests(unittest.IsolatedAsyncioTestCase):
    """Planner 自主决策、温度滑块门控与纯随机概率模式测试。"""

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

    @staticmethod
    def _sample_planner_tools() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "reply",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chat_anlyze": {"type": "string"},
                            "plugin_options": {
                                "type": "object",
                                "properties": {
                                    "custom.maibot-fishaudio-voice-pro.voice_reply": {"type": "object"},
                                    "other.plugin.extra": {"type": "object"},
                                },
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {"name": "send_voice_reply", "parameters": {"type": "object", "properties": {}}},
            },
        ]

    async def test_planner_before_request_injection_and_non_planner_filtering(self) -> None:
        plugin = self._build_plugin(auto_voice={"mode": "planner", "planner_temperature": 0.25})
        items = [
            {"item_type": "SystemMessageItem", "meta": {"item_id": "sys_0"}, "parts": [{"type": "text", "text": "sys"}]},
            {"item_type": "UserMessageItem", "meta": {"item_id": "u_1"}, "parts": [{"type": "text", "text": "hello"}]},
        ]
        tools = self._sample_planner_tools()

        res = await plugin.planner_before_request_hook(
            items=items,
            tool_definitions=tools,
            session_id="qq_1001",
        )
        assert res is not None
        self.assertEqual(res["action"], "continue")
        new_items = res["modified_kwargs"]["items"]
        self.assertEqual(len(new_items), 3)
        self.assertEqual(new_items[1]["meta"]["item_id"], "fishaudio-voice-planner-qq_1001")
        self.assertIn("0.25", new_items[1]["parts"][0]["text"])
        self.assertIn("低频克制", new_items[1]["parts"][0]["text"])

        # 切换到 off / probability / always 后应移除注入消息、隐藏 send_voice_reply 及 reply 中的语音扩展
        plugin.config.auto_voice.mode = "probability"
        res_off = await plugin.planner_before_request_hook(
            items=new_items,
            tool_definitions=tools,
            session_id="qq_1001",
        )
        assert res_off is not None
        cleaned_items = res_off["modified_kwargs"]["items"]
        self.assertEqual(len(cleaned_items), 2)
        filtered_tools = res_off["modified_kwargs"]["tool_definitions"]
        self.assertEqual([t["function"]["name"] for t in filtered_tools], ["reply"])
        reply_plugin_props = filtered_tools[0]["function"]["parameters"]["properties"]["plugin_options"]["properties"]
        self.assertNotIn("custom.maibot-fishaudio-voice-pro.voice_reply", reply_plugin_props)
        self.assertIn("other.plugin.extra", reply_plugin_props)

    async def test_voice_reply_extension_temperature_gating_and_transform(self) -> None:
        plugin = self._build_plugin(auto_voice={"mode": "planner", "planner_temperature": 0.3, "max_segment_chars": 80})

        async def fake_synth(text: str, **kwargs: Any) -> dict[str, Any]:
            raw = f"VOICE:{text}".encode("utf-8")
            return {
                "success": True,
                "audio_base64": base64.b64encode(raw).decode("ascii"),
                "audio_bytes_len": len(raw),
                "text": text,
            }

        plugin._tts_service.synthesize = AsyncMock(side_effect=fake_synth)  # noqa: SLF001

        # prepare 必须返回空字典，确保兼容 passive_replyer 与 active reply 工具两种模式
        self.assertEqual(await plugin.voice_reply_extension(phase="prepare", parameters={}), {})

        sample_messages = [
            {"segments": [{"type": "text", "data": "[laughing] 早安呀。今天开心吗？"}], "quote_previous": True},
            {"segments": [{"type": "emoji", "data": "笑", "hash": "abc", "binary_data_base64": "AAAA"}], "quote_previous": False},
        ]

        # 温度 0.3 要求 suitability >= 0.7；suitability=0.5 且非明确要求应被过滤并剥离舞台标签保留文字
        res_filtered = await plugin.voice_reply_extension(
            phase="before_send",
            parameters={"use_voice": True, "explicit_request": False, "suitability": 0.5},
            messages=sample_messages,
        )
        self.assertIn("messages", res_filtered)
        self.assertEqual(res_filtered["messages"][0]["segments"], [{"type": "text", "data": "早安呀。今天开心吗？"}])
        plugin._tts_service.synthesize.assert_not_awaited()  # noqa: SLF001

        # 温度降至 0.0 时，若 explicit_request=True（用户明确要求发语音）仍可直接放行
        plugin.config.auto_voice.planner_temperature = 0.0
        res_explicit = await plugin.voice_reply_extension(
            phase="before_send",
            parameters={"use_voice": True, "explicit_request": True, "suitability": 0.2, "style": "gentle"},
            messages=sample_messages,
        )
        self.assertIn("messages", res_explicit)
        out_msgs = res_explicit["messages"]
        self.assertEqual(len(out_msgs), 3)
        self.assertEqual(out_msgs[0]["segments"][0]["type"], "voice")
        self.assertEqual(out_msgs[0]["segments"][0]["data"], "[gentle] [laughing] 早安呀。")
        self.assertFalse(out_msgs[0]["quote_previous"])
        self.assertEqual(out_msgs[1]["segments"][0]["type"], "voice")
        self.assertEqual(out_msgs[1]["segments"][0]["data"], "[gentle] 今天开心吗？")
        # 表情附件消息必须原样保留
        self.assertEqual(out_msgs[2]["segments"][0]["type"], "emoji")

        # 验证输出格式符合 MaiBot ReplyExtension 协议契约
        for msg in out_msgs:
            self.assertEqual(set(msg.keys()), {"segments", "quote_previous"})
            self.assertIs(type(msg["quote_previous"]), bool)
            self.assertTrue(isinstance(msg["segments"], list) and len(msg["segments"]) > 0)
            for seg in msg["segments"]:
                if seg["type"] == "voice":
                    self.assertIsInstance(seg["data"], str)
                    decoded = base64.b64decode(seg["binary_data_base64"], validate=True)
                    self.assertTrue(len(decoded) > 0)

        # 局部合成失败时：已合成段发语音，未合成段清洗标签后保留文字
        plugin.config.auto_voice.planner_temperature = 0.8
        plugin._tts_service.synthesize = AsyncMock(  # noqa: SLF001
            side_effect=[
                {"success": True, "audio_base64": base64.b64encode(b"OK").decode("ascii"), "audio_bytes_len": 2, "text": "第一句。"},
                {"success": False, "error": "upstream 503"},
            ]
        )
        res_partial = await plugin.voice_reply_extension(
            phase="before_send",
            parameters={"use_voice": True, "suitability": 0.6},
            messages=[{"segments": [{"type": "text", "data": "第一句。[sighs] 第二句。"}], "quote_previous": True}],
        )
        self.assertEqual(len(res_partial["messages"]), 2)
        self.assertEqual(res_partial["messages"][0]["segments"][0]["type"], "voice")
        self.assertEqual(res_partial["messages"][1]["segments"], [{"type": "text", "data": "第二句。"}])
        self.assertTrue(res_partial["messages"][1]["quote_previous"])

        # 关闭文字降级且首段失败时应显式抛错交由 Replyer/reply 链路处理
        plugin.config.fallback.fallback_to_text = False
        plugin._tts_service.synthesize = AsyncMock(return_value={"success": False, "error": "tts down"})  # noqa: SLF001
        with self.assertRaises(RuntimeError):
            await plugin.voice_reply_extension(
                phase="before_send",
                parameters={"use_voice": True, "suitability": 0.9},
                messages=[{"segments": [{"type": "text", "data": "测试失败不退回"}], "quote_previous": False}],
            )

    async def test_probability_mode_sampling_ttl_and_voice_quote_guard(self) -> None:
        plugin = self._build_plugin(
            auto_voice={"mode": "probability", "random_probability": 0.4, "decision_ttl_seconds": 8.0}
        )
        plugin._tts_service.synthesize = AsyncMock(  # noqa: SLF001
            return_value={
                "success": True,
                "audio_base64": base64.b64encode(b"PROB_MP3").decode("ascii"),
                "audio_bytes_len": 8,
                "text": "随机语音测试",
            }
        )

        # 首次掷骰 0.2 < 0.4 -> 命中语音；第二次仍在 TTL 内即使 random 返回 0.9 也复用缓存结果
        with patch.object(plugin_mod.random, "random", side_effect=[0.2, 0.9]):
            r1 = await plugin.auto_voice_hook(
                message={"session_id": "s_prob", "raw_message": [{"type": "text", "data": "随机语音测试"}]},
                set_reply=False,
            )
            r2 = await plugin.auto_voice_hook(
                message={"session_id": "s_prob", "raw_message": [{"type": "text", "data": "随机语音测试"}]},
                set_reply=False,
            )
        assert r1 is not None and r2 is not None
        self.assertEqual(r1["modified_kwargs"]["message"]["raw_message"][0]["type"], "voice")
        self.assertEqual(r2["modified_kwargs"]["message"]["raw_message"][0]["type"], "voice")

        # 热重载清空决策缓存且 random=0.9 >= 0.4 -> 不转语音放行原文
        await plugin.on_config_update(scope="self")
        with patch.object(plugin_mod.random, "random", return_value=0.9):
            r3 = await plugin.auto_voice_hook(
                message={"session_id": "s_prob", "raw_message": [{"type": "text", "data": "保持文字"}]},
                set_reply=False,
            )
        self.assertIsNone(r3)

        # planner 模式下：普通文字消息直接放行；已有 voice 段且带 set_reply=True 时自动清掉引用标记保护投递
        plugin.config.auto_voice.mode = "planner"
        self.assertIsNone(
            await plugin.auto_voice_hook(
                message={"session_id": "s_p", "raw_message": [{"type": "text", "data": "Planner 决定发文字"}]},
                set_reply=True,
            )
        )
        voice_guard = await plugin.auto_voice_hook(
            message={"session_id": "s_p", "raw_message": [{"type": "voice", "data": "hi", "binary_data_base64": "AAAA"}]},
            set_reply=True,
            reply_message_id="m_123",
        )
        assert voice_guard is not None
        self.assertFalse(voice_guard["modified_kwargs"]["set_reply"])
        self.assertEqual(voice_guard["modified_kwargs"]["reply_message_id"], "")

    async def test_mode_and_slider_boundary_conditions(self) -> None:
        """测试 4 种触发模式与双滑块 0.0/0.5/1.0 边界值及 explicit_request=True。"""
        # 1. _should_allow_planner_voice 算法各温度与契合度、显式要求边界
        should_allow = plugin_mod._should_allow_planner_voice

        # 1.1 温度 0.0（极度克制）：非明确要求一律为 False，即便契合度 1.0 也必须被拦截
        self.assertFalse(should_allow(0.0, explicit_request=False, suitability=1.0))
        self.assertFalse(should_allow(0.0, explicit_request=False, suitability=0.0))
        # 但 explicit_request=True 时必须无条件放行，哪怕 suitability=0.0
        self.assertTrue(should_allow(0.0, explicit_request=True, suitability=0.0))

        # 1.2 温度 1.0（极高倾向）：只要模型调用语音扩展，哪怕 suitability=0.0 也放行
        self.assertTrue(should_allow(1.0, explicit_request=False, suitability=0.0))
        self.assertTrue(should_allow(1.0, explicit_request=True, suitability=0.0))

        # 1.3 温度 0.5（均衡自然）：门槛阈值为 1.0 - 0.5 = 0.5
        self.assertTrue(should_allow(0.5, explicit_request=False, suitability=0.5))
        self.assertTrue(should_allow(0.5, explicit_request=False, suitability=0.51))
        self.assertFalse(should_allow(0.5, explicit_request=False, suitability=0.49))
        self.assertTrue(should_allow(0.5, explicit_request=True, suitability=0.1))

        # 1.4 越界温度自动归一化 clamp ( <0.0 按 0.0 处理, >1.0 按 1.0 处理 )
        self.assertFalse(should_allow(-0.5, explicit_request=False, suitability=1.0))
        self.assertTrue(should_allow(-0.5, explicit_request=True, suitability=0.0))
        self.assertTrue(should_allow(1.5, explicit_request=False, suitability=0.0))

        # 2. _should_convert_by_probability 随机概率滑块边界
        plugin = self._build_plugin()
        # prob=0.0: 决策必须恒为 False，且不产生缓存干扰
        dec_0, _ = plugin._should_convert_by_probability("s_zero", 0.0, ttl_seconds=0.0)
        self.assertFalse(dec_0)
        # prob=1.0: 决策必须恒为 True
        dec_1, _ = plugin._should_convert_by_probability("s_one", 1.0, ttl_seconds=0.0)
        self.assertTrue(dec_1)
        # 越界 clamp 测试
        dec_neg, _ = plugin._should_convert_by_probability("s_neg", -0.2, ttl_seconds=0.0)
        self.assertFalse(dec_neg)
        dec_over, _ = plugin._should_convert_by_probability("s_over", 1.5, ttl_seconds=0.0)
        self.assertTrue(dec_over)

        # 3. 4 种触发模式在 send_service.before_send (auto_voice_hook) 下的行为
        async def fake_synth(text: str, **kwargs: Any) -> dict[str, Any]:
            raw = f"VOICE:{text}".encode("utf-8")
            return {
                "success": True,
                "audio_base64": base64.b64encode(raw).decode("ascii"),
                "audio_bytes_len": len(raw),
                "text": text,
            }

        plugin._tts_service.synthesize = AsyncMock(side_effect=fake_synth)
        sample_msg = {"session_id": "stream_mode_test", "raw_message": [{"type": "text", "data": "测试四种模式表现"}]}

        # 3.1 mode="off": auto_voice_hook 必须直接返回 None（不转语音）
        plugin.config.auto_voice.mode = "off"
        self.assertIsNone(
            await plugin.auto_voice_hook(
                message={"session_id": "stream_off", "raw_message": [{"type": "text", "data": "测试四种模式表现"}]},
                set_reply=False,
            )
        )

        # 3.2 mode="planner": auto_voice_hook 必须直接返回 None（交由 ReplyExtension 处理，不盲转）
        plugin.config.auto_voice.mode = "planner"
        self.assertIsNone(
            await plugin.auto_voice_hook(
                message={"session_id": "stream_planner", "raw_message": [{"type": "text", "data": "测试四种模式表现"}]},
                set_reply=False,
            )
        )

        # 3.3 mode="probability" 且 random_probability=0.0: 恒不转语音
        plugin.config.auto_voice.mode = "probability"
        plugin.config.auto_voice.random_probability = 0.0
        self.assertIsNone(
            await plugin.auto_voice_hook(
                message={"session_id": "stream_prob_0", "raw_message": [{"type": "text", "data": "测试四种模式表现"}]},
                set_reply=False,
            )
        )

        # 3.4 mode="probability" 且 random_probability=1.0: 恒转语音
        plugin.config.auto_voice.random_probability = 1.0
        res_prob_1 = await plugin.auto_voice_hook(
            message={"session_id": "stream_prob_1", "raw_message": [{"type": "text", "data": "测试四种模式表现"}]},
            set_reply=False,
        )
        assert res_prob_1 is not None
        self.assertEqual(res_prob_1["modified_kwargs"]["message"]["raw_message"][0]["type"], "voice")

        # 3.5 mode="always": 恒转语音
        plugin.config.auto_voice.mode = "always"
        res_always = await plugin.auto_voice_hook(
            message={"session_id": "stream_always", "raw_message": [{"type": "text", "data": "测试四种模式表现"}]},
            set_reply=False,
        )
        assert res_always is not None
        self.assertEqual(res_always["modified_kwargs"]["message"]["raw_message"][0]["type"], "voice")

    async def test_voice_reply_extension_non_planner_and_deferred_temp_cleanup(self) -> None:
        """测试 voice_reply 扩展在非 planner 模式下的文字保全，以及延迟临时文件清理机制。"""
        # 1. 在非 planner 模式（off, always, probability）下，voice_reply_extension 不应合成语音，应剥离标签保留文字
        sample_messages = [
            {"segments": [{"type": "text", "data": "[whispering] 这是私聊悄悄话。"}], "quote_previous": True}
        ]
        for non_planner_mode in ("off", "always", "probability"):
            plugin_np = self._build_plugin(auto_voice={"mode": non_planner_mode})
            plugin_np._tts_service.synthesize = AsyncMock()
            res_np = await plugin_np.voice_reply_extension(
                phase="before_send",
                parameters={"use_voice": True, "suitability": 1.0},
                messages=sample_messages,
            )
            plugin_np._tts_service.synthesize.assert_not_awaited()
            self.assertIn("messages", res_np)
            self.assertEqual(res_np["messages"][0]["segments"][0]["data"], "这是私聊悄悄话。")

        # 2. 延迟临时文件清理 (defer_immediate_cleanup)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            plugin_storage = self._build_plugin(
                auto_voice={"mode": "planner", "planner_temperature": 0.8},
                storage={
                    "save_output_audio": True,
                    "output_dir": str(out_dir),
                    "auto_delete": True,
                    "retain_minutes": 0,
                    "max_saved_files": 100,
                },
            )

            async def fake_synth_storage(text: str, **kwargs: Any) -> dict[str, Any]:
                return {
                    "success": True,
                    "audio_base64": base64.b64encode(b"TEMP_VOICE_PAYLOAD").decode("ascii"),
                    "audio_bytes_len": 18,
                    "text": text,
                }

            plugin_storage._tts_service.synthesize = AsyncMock(side_effect=fake_synth_storage)
            res_ext = await plugin_storage.voice_reply_extension(
                phase="before_send",
                parameters={"use_voice": True, "suitability": 0.9},
                messages=[{"segments": [{"type": "text", "data": "测试延迟临时文件清理。"}], "quote_previous": False}],
            )
            self.assertEqual(res_ext["messages"][0]["segments"][0]["type"], "voice")
            # 确认 finally 块已触发 _cleanup_deferred_files，临时落盘的文件已全部被清除
            leftover = [p for p in out_dir.iterdir() if p.name.startswith("fishaudio_")]
            self.assertEqual(leftover, [])

    async def test_planner_before_request_non_planner_modes_and_standalone_option_cleanup(self) -> None:
        """测试 planner_before_request 在 off/always 模式下的清理与 reply plugin_options 独占清理。"""
        # 当 reply 工具的 plugin_options 仅有 voice_reply 这一个扩展时，清理后应彻底移除 plugin_options 键
        single_ext_tools = [
            {
                "type": "function",
                "function": {
                    "name": "reply",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chat_anlyze": {"type": "string"},
                            "plugin_options": {
                                "type": "object",
                                "properties": {
                                    "custom.maibot-fishaudio-voice-pro.voice_reply": {"type": "object"},
                                },
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {"name": "send_voice_reply", "parameters": {"type": "object", "properties": {}}},
            },
        ]
        items = [
            {"item_type": "SystemMessageItem", "meta": {"item_id": "fishaudio-voice-planner-1"}, "parts": [{"type": "text", "text": "old"}]},
            {"item_type": "UserMessageItem", "meta": {"item_id": "u1"}, "parts": [{"type": "text", "text": "hi"}]},
        ]

        # mode="off": 移除注入项，隐藏 send_voice_reply，且 plugin_options 被整键 pop
        p_off = self._build_plugin(auto_voice={"mode": "off"})
        res_off = await p_off.planner_before_request_hook(items=items, tool_definitions=single_ext_tools, session_id="s_off")
        assert res_off is not None
        mod_tools = res_off["modified_kwargs"]["tool_definitions"]
        self.assertEqual(len(mod_tools), 1)
        self.assertEqual(mod_tools[0]["function"]["name"], "reply")
        self.assertNotIn("plugin_options", mod_tools[0]["function"]["parameters"]["properties"])
        mod_items = res_off["modified_kwargs"]["items"]
        self.assertEqual(len(mod_items), 1)
        self.assertEqual(mod_items[0]["meta"]["item_id"], "u1")

        # mode="always": 行为与 off 一致（出站由 before_send 处理，Planner 不应调用扩展或工具）
        p_always = self._build_plugin(auto_voice={"mode": "always"})
        res_always = await p_always.planner_before_request_hook(items=items, tool_definitions=single_ext_tools, session_id="s_alw")
        assert res_always is not None
        mod_tools_alw = res_always["modified_kwargs"]["tool_definitions"]
        self.assertNotIn("plugin_options", mod_tools_alw[0]["function"]["parameters"]["properties"])

        # 检查温度档位描述生成 0.0, 0.5, 1.0
        p_planner = self._build_plugin(auto_voice={"mode": "planner", "planner_temperature": 0.0})
        res_t0 = await p_planner.planner_before_request_hook(items=[{"item_type": "UserMessageItem", "meta": {"item_id": "u1"}, "parts": []}], tool_definitions=[], session_id="s0")
        self.assertIn("极度克制（仅明确要求）", res_t0["modified_kwargs"]["items"][0]["parts"][0]["text"])

        p_planner.config.auto_voice.planner_temperature = 0.5
        res_t5 = await p_planner.planner_before_request_hook(items=[{"item_type": "UserMessageItem", "meta": {"item_id": "u1"}, "parts": []}], tool_definitions=[], session_id="s5")
        self.assertIn("均衡自然", res_t5["modified_kwargs"]["items"][0]["parts"][0]["text"])

        p_planner.config.auto_voice.planner_temperature = 1.0
        res_t10 = await p_planner.planner_before_request_hook(items=[{"item_type": "UserMessageItem", "meta": {"item_id": "u1"}, "parts": []}], tool_definitions=[], session_id="s10")
        self.assertIn("极高倾向", res_t10["modified_kwargs"]["items"][0]["parts"][0]["text"])

    async def test_on_config_update_three_args_and_reported_plugin_options(self) -> None:
        """验证热重载兼容宿主 3 参数调用，且 suitability=0.85/use_voice=True 正常触发语音并记录跳过原因。"""
        plugin = self._build_plugin(auto_voice={"mode": "planner", "planner_temperature": 0.5})
        old_service = plugin._tts_service  # noqa: SLF001
        assert old_service is not None
        old_service.close = AsyncMock()

        # 模拟 MaiBot Runner 传入 (scope, config_data, config_version) 三个位置参数
        await plugin.on_config_update("self", {"auto_voice": {"mode": "planner", "planner_temperature": 0.9}}, "1.1.0")
        old_service.close.assert_awaited_once()
        self.assertIsNot(plugin._tts_service, old_service)  # noqa: SLF001

        plugin._tts_service.synthesize = AsyncMock(  # noqa: SLF001
            return_value={
                "success": True,
                "audio_base64": base64.b64encode(b"OK_085").decode("ascii"),
                "audio_bytes_len": 6,
                "text": "多看两眼怎么了",
            }
        )
        # 复现用户上报的参数组合（不含 voice 键，explicit_request=False, style="", suitability=0.85, use_voice=True）
        res_085 = await plugin.voice_reply_extension(
            phase="before_send",
            parameters={
                "explicit_request": False,
                "style": "",
                "suitability": 0.85,
                "use_voice": True,
            },
            messages=[{"segments": [{"type": "text", "data": "多看两眼怎么了"}], "quote_previous": True}],
        )
        self.assertEqual(res_085["messages"][0]["segments"][0]["type"], "voice")
        self.assertFalse(res_085["messages"][0]["quote_previous"])

        # 含 URL 时应跳过转语音并记录清晰原因日志
        with self.assertLogs("fishaudio.test", level="INFO") as captured:
            res_url = await plugin.voice_reply_extension(
                phase="before_send",
                parameters={"explicit_request": False, "style": "", "suitability": 0.85, "use_voice": True},
                messages=[{"segments": [{"type": "text", "data": "[laughing] 看这里 https://example.com"}], "quote_previous": True}],
            )
        self.assertEqual(res_url["messages"][0]["segments"][0]["data"], "看这里 https://example.com")
        self.assertTrue(any("包含跳过模式" in msg for msg in captured.output))


class EmotionInjectionTests(unittest.IsolatedAsyncioTestCase):
    """Fish Audio 行内情绪与语气参数自动注入单元测试。"""

    SAMPLE_PENGUIN_TEXT = (
        "嘿嘿，快看这里！如果你现在还没关注臭企鹅，那你的肚子肯定会变得圆滚滚的！"
        "圆圆滚滚！只有每天准时收看我的视频，才能变成最可爱的企鹅。"
        "要是错过了，我就要对着你滚滚滚，滚滚开开！嘿咻嘿咻~"
    )
    EXPECTED_PENGUIN_ANNOTATED = (
        "[开心的语气]嘿嘿，快看这里！[平静]如果你现在还没关注臭企鹅，那你的肚子肯定会变得圆滚滚的！"
        "[语气激动]圆圆滚滚！[平静]只有每天准时收看我的视频，才能变成最可爱的企鹅。"
        "要是错过了，我就要对着你[语气激动]滚滚滚，滚滚开开！[哼歌]嘿咻嘿咻~"
    )

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

    def test_penguin_example_rule_injection_and_zero_tampering(self) -> None:
        injected = inject_emotion_tags_by_rules(self.SAMPLE_PENGUIN_TEXT, model="s2.1-pro-free")
        self.assertEqual(injected, self.EXPECTED_PENGUIN_ANNOTATED)
        self.assertEqual(strip_audio_tags(injected), self.SAMPLE_PENGUIN_TEXT)

        # s1 模型自动切换为圆括号标签语法
        self.assertEqual(format_emotion_tag("开心的语气", model="s1"), "(开心的语气)")
        injected_s1 = inject_emotion_tags_by_rules("嘿嘿，早安！", model="s1")
        self.assertEqual(injected_s1, "(开心的语气)嘿嘿，早安！")

    def test_reconcile_injected_text_guards_against_llm_tampering(self) -> None:
        # 1. 带代码块围栏、引导前缀与标签后多余空格时清洗还原原文
        noisy_cand = (
            "```text\n"
            "注入后：[开心的语气] 嘿嘿，快看这里！[平静] 如果你现在还没关注臭企鹅。\n"
            "```"
        )
        orig = "嘿嘿，快看这里！如果你现在还没关注臭企鹅。"
        reconciled = reconcile_injected_text(orig, noisy_cand, model="s2.1-pro-free")
        self.assertEqual(reconciled, "[开心的语气]嘿嘿，快看这里！[平静]如果你现在还没关注臭企鹅。")

        # 2. LLM 漏字/改字时自动对齐还原原始字词并保留正确位置标签
        tampered_cand = "[平静]要是错过了，我就对着你[语气激动]滚滚滚！"
        orig2 = "要是错过了，我就要对着你滚滚滚！"
        reconciled2 = reconcile_injected_text(orig2, tampered_cand, model="s2.1-pro-free")
        self.assertEqual(reconciled2, "[平静]要是错过了，我就要对着你[语气激动]滚滚滚！")
        self.assertEqual(strip_audio_tags(reconciled2 or ""), orig2)

        # 3. 无关幻觉或无标签时返回 None
        self.assertIsNone(reconcile_injected_text(orig2, "[开心]完全不相关的一段新台词", model="s2.1-pro-free"))
        self.assertIsNone(reconcile_injected_text(orig2, orig2, model="s2.1-pro-free"))

    async def test_emotion_modes_timeout_fallback_and_skip_tagged(self) -> None:
        # 1. hybrid 模式：LLM 成功时直接采用 LLM 结果，并透传 task_name
        plugin = self._build_plugin(
            emotion={"enabled": True, "mode": "hybrid", "model_task": "utils", "timeout_seconds": 2.0}
        )
        plugin.ctx.llm.generate = AsyncMock(
            return_value={"success": True, "response": "[温柔]晚安呀，[轻笑]记得盖好被子。"}
        )
        res_llm = await plugin.inject_emotion_tags("晚安呀，记得盖好被子。")
        self.assertEqual(res_llm, "[温柔]晚安呀，[轻笑]记得盖好被子。")
        self.assertEqual(plugin.ctx.llm.generate.await_args.kwargs.get("task_name"), "utils")

        # 2. 已包含行内标签时跳过重复注入
        self.assertTrue(has_inline_emotion_tags("[laughing] 早安"))
        plugin.ctx.llm.generate.reset_mock()
        self.assertEqual(await plugin.inject_emotion_tags("[laughing] 早安"), "[laughing] 早安")
        plugin.ctx.llm.generate.assert_not_awaited()

        # 3. hybrid 模式：LLM 超时后无感回退本地规则注入
        plugin.ctx.llm.generate = AsyncMock(side_effect=asyncio.TimeoutError())
        fallback_rule = await plugin.inject_emotion_tags(self.SAMPLE_PENGUIN_TEXT)
        self.assertEqual(fallback_rule, self.EXPECTED_PENGUIN_ANNOTATED)

        # 4. llm 模式：LLM 失败时保留原文
        plugin.config.emotion.mode = "llm"
        plugin.ctx.llm.generate = AsyncMock(return_value={"success": False, "error": "no model"})
        self.assertEqual(await plugin.inject_emotion_tags("嘿嘿，你好！"), "嘿嘿，你好！")

        # 5. rule 模式：绝不调用 LLM
        plugin.config.emotion.mode = "rule"
        plugin.ctx.llm.generate = AsyncMock()
        self.assertEqual(await plugin.inject_emotion_tags("嘿嘿，你好！"), "[开心的语气]嘿嘿，你好！")
        plugin.ctx.llm.generate.assert_not_awaited()

        # 6. 关闭开关时完全透传原文
        plugin.config.emotion.enabled = False
        self.assertEqual(await plugin.inject_emotion_tags(self.SAMPLE_PENGUIN_TEXT), self.SAMPLE_PENGUIN_TEXT)

    async def test_multi_segment_global_injection_and_clean_history_isolation(self) -> None:
        plugin = self._build_plugin(
            auto_voice={"mode": "always", "max_segment_chars": 70, "segment_delay_ms": 0},
            emotion={"enabled": True, "mode": "hybrid"},
        )
        plugin.ctx.llm.generate = AsyncMock(
            return_value={
                "success": True,
                "response": (
                    "[开心的语气]嘿嘿，快看这里！\n"
                    "[平静]如果你现在还没关注臭企鹅，那你的肚子肯定会变得圆滚滚的！\n"
                    "[语气激动]圆圆滚滚！\n"
                    "[平静]只有每天准时收看我的视频，才能变成最可爱的企鹅。\n"
                    "要是错过了，我就要对着你[语气激动]滚滚滚，滚滚开开！\n"
                    "[哼歌]嘿咻嘿咻~"
                ),
            }
        )

        sent_texts: list[str] = []

        async def capture_synth(text: str, **kwargs: Any) -> dict[str, Any]:
            sent_texts.append(text)
            raw = f"AUDIO:{text}".encode("utf-8")
            return {
                "success": True,
                "audio_base64": base64.b64encode(raw).decode("ascii"),
                "audio_bytes_len": len(raw),
                "text": text,
            }

        plugin._tts_service.synthesize = AsyncMock(side_effect=capture_synth)  # noqa: SLF001

        msg = {
            "session_id": "group_penguin",
            "raw_message": [{"type": "text", "data": self.SAMPLE_PENGUIN_TEXT}],
        }
        res = await plugin.auto_voice_hook(message=msg, set_reply=False)
        assert res is not None

        # 多段合成只调 1 次 LLM 全局标注
        plugin.ctx.llm.generate.assert_awaited_once()
        self.assertGreaterEqual(len(sent_texts), 2)
        joined_sent = "".join(sent_texts)
        self.assertIn("[开心的语气]", joined_sent)
        self.assertIn("[平静]", joined_sent)
        self.assertIn("[语气激动]", joined_sent)
        self.assertIn("[哼歌]", joined_sent)
        # 相邻接续分段若原句首无标签（如“要是错过了...”独立成段时）应自动继承前一段活跃情绪 [平静]
        self.assertTrue(all(t.startswith("[") for t in sent_texts))

        # 关闭展示开关时 voice.data 仅保留原文；默认开启时 voice.data 直接展示带标签的转换文本
        final_voice_seg = res["modified_kwargs"]["message"]["raw_message"][0]
        self.assertEqual(final_voice_seg["type"], "voice")
        self.assertIn("[哼歌]", final_voice_seg["data"])
        self.assertIn("嘿咻嘿咻~", final_voice_seg["data"])

        # 关闭 show_injected_text 后，写回消息段的 data 保持干净原文，不含情绪标签
        plugin.config.emotion.show_injected_text = False
        plugin.ctx.llm.generate.reset_mock()
        res_clean = await plugin.auto_voice_hook(message=msg, set_reply=False)
        assert res_clean is not None
        clean_voice_seg = res_clean["modified_kwargs"]["message"]["raw_message"][0]
        self.assertNotIn("[", clean_voice_seg["data"])
        self.assertIn("嘿咻嘿咻~", clean_voice_seg["data"])

        # api_synthesize 单句直调也默认注入标签并返回 injected_text，同时保持 text 为原文
        plugin.config.emotion.mode = "rule"
        api_res = await plugin.api_synthesize("嘿嘿，快看这里！")
        self.assertTrue(api_res["success"])
        self.assertEqual(api_res["text"], "嘿嘿，快看这里！")
        self.assertEqual(api_res["injected_text"], "[开心的语气]嘿嘿，快看这里！")

    async def test_info_logging_and_send_injected_text_to_chat_preview(self) -> None:
        # 1. 验证 inject_emotion_tags 与 FishAudioTTSService.synthesize 均输出带情绪标签的 INFO 日志
        plugin = self._build_plugin(
            auto_voice={"mode": "always", "segment_delay_ms": 0},
            emotion={"enabled": True, "mode": "rule", "show_injected_text": True, "send_injected_text_to_chat": True},
        )
        with self.assertLogs("fishaudio.test", level="INFO") as captured_plugin:
            injected = await plugin.inject_emotion_tags("嘿嘿，快看这里！")
        self.assertEqual(injected, "[开心的语气]嘿嘿，快看这里！")
        self.assertTrue(any("[开心的语气]嘿嘿，快看这里！" in m for m in captured_plugin.output))

        svc = FishAudioTTSService("sk-test-1234", logger=plugin.ctx.logger)
        fake_session = MagicMock()
        fake_session.closed = False
        fake_session.post = MagicMock(return_value=_FakeResponse(status=200, body=b"MP3DATA"))
        fake_session.close = AsyncMock()
        svc._session = fake_session  # noqa: SLF001
        with self.assertLogs("fishaudio.test", level="INFO") as captured_svc:
            synth_res = await svc.synthesize("[开心的语气]嘿嘿，快看这里！")
        self.assertTrue(synth_res["success"])
        self.assertTrue(any("text=[开心的语气]嘿嘿，快看这里！" in m for m in captured_svc.output))
        await svc.close()

        # 2. 验证 send_injected_text_to_chat=True 在 auto_voice_hook 下发出带标签预览，且不会引发二次语音环回
        async def fake_synth(text: str, **kwargs: Any) -> dict[str, Any]:
            raw = f"AUDIO:{text}".encode("utf-8")
            return {
                "success": True,
                "audio_base64": base64.b64encode(raw).decode("ascii"),
                "audio_bytes_len": len(raw),
                "text": text,
            }

        plugin._tts_service.synthesize = AsyncMock(side_effect=fake_synth)  # noqa: SLF001
        res_auto = await plugin.auto_voice_hook(
            message={"session_id": "s_preview", "raw_message": [{"type": "text", "data": "嘿嘿，快看这里！"}]},
            set_reply=False,
        )
        assert res_auto is not None
        plugin.ctx.send.text.assert_awaited_once_with("[开心的语气]嘿嘿，快看这里！", "s_preview")
        # 预览消息再次进入 auto_voice_hook 时应命中 bypass 令牌而直接放行，不会再转成语音
        bypassed = await plugin.auto_voice_hook(
            message={"session_id": "s_preview", "raw_message": [{"type": "text", "data": "[开心的语气]嘿嘿，快看这里！"}]},
            set_reply=False,
        )
        self.assertIsNone(bypassed)

        # 3. 验证 send_voice_reply 与 voice_reply_extension 均支持 send_injected_text_to_chat
        plugin.ctx.send.text.reset_mock()
        res_tool = await plugin.send_voice_reply(text="嘿嘿，快看这里！", stream_id="s_tool")
        self.assertTrue(res_tool["success"])
        plugin.ctx.send.text.assert_awaited_once_with("[开心的语气]嘿嘿，快看这里！", "s_tool")

        plugin.config.auto_voice.mode = "planner"
        plugin.config.auto_voice.planner_temperature = 0.8
        res_ext = await plugin.voice_reply_extension(
            phase="before_send",
            parameters={"use_voice": True, "suitability": 0.9},
            messages=[{"segments": [{"type": "text", "data": "嘿嘿，快看这里！"}], "quote_previous": False}],
        )
        self.assertEqual(len(res_ext["messages"]), 2)
        self.assertEqual(res_ext["messages"][0]["segments"][0]["type"], "voice")
        self.assertEqual(res_ext["messages"][0]["segments"][0]["data"], "[开心的语气]嘿嘿，快看这里！")
        self.assertEqual(res_ext["messages"][1]["segments"], [{"type": "text", "data": "[开心的语气]嘿嘿，快看这里！"}])

    def test_split_tts_segments_and_map_annotated_preserve_tags(self) -> None:
        # 硬切分超长带标签文本时不截断 [标签]
        tagged_long = "[开心的语气]一二三四五六七八九十甲乙丙丁戊己庚辛"
        segs = split_tts_segments(tagged_long, max_chars=15, split_sentences=True)
        self.assertGreaterEqual(len(segs), 2)
        self.assertTrue(segs[0].startswith("[开心的语气]"))
        self.assertFalse(any("[" in s and "]" not in s for s in segs))

        # 跨段映射能把上一段末尾活跃标签延续到下一段句首
        mapped = map_annotated_to_segments(
            ["只有每天准时收看我的视频，才能变成最可爱的企鹅。", "要是错过了，我就要对着你滚滚滚，滚滚开开！"],
            "[平静]只有每天准时收看我的视频，才能变成最可爱的企鹅。\n要是错过了，我就要对着你[语气激动]滚滚滚，滚滚开开！",
        )
        self.assertEqual(
            mapped,
            [
                "[平静]只有每天准时收看我的视频，才能变成最可爱的企鹅。",
                "[平静]要是错过了，我就要对着你[语气激动]滚滚滚，滚滚开开！",
            ],
        )

    async def test_fishaudio_command_normal_and_force_emo(self) -> None:
        import re as _re

        plugin = self._build_plugin(
            auto_voice={"mode": "off", "segment_delay_ms": 0},
            emotion={"enabled": False, "mode": "rule", "show_injected_text": True},
        )
        cmd_comp = {c["name"]: c for c in plugin.get_components()}["fishaudio"]
        cmd_re = _re.compile(cmd_comp["metadata"]["command_pattern"])

        sent_tts: list[str] = []

        async def capture_synth(text: str, **kwargs: Any) -> dict[str, Any]:
            sent_tts.append(text)
            raw = f"CMD_AUDIO:{text}".encode("utf-8")
            return {
                "success": True,
                "audio_base64": base64.b64encode(raw).decode("ascii"),
                "audio_bytes_len": len(raw),
                "text": text,
            }

        plugin._tts_service.synthesize = AsyncMock(side_effect=capture_synth)  # noqa: SLF001

        # 1. 空参数 (/fishaudio 和 /fishaudio emo) 发出用法提示且不调用 TTS
        for raw_empty in ("/fishaudio", "/fishaudio emo"):
            m_empty = cmd_re.search(raw_empty)
            assert m_empty is not None
            ok_empty, _, intercept_empty = await plugin.handle_fishaudio_command(
                text=raw_empty,
                matched_groups=m_empty.groupdict(),
                stream_id="cmd_s1",
            )
            self.assertFalse(ok_empty)
            self.assertTrue(intercept_empty)
        self.assertEqual(plugin.ctx.send.text.await_count, 2)
        plugin._tts_service.synthesize.assert_not_awaited()  # noqa: SLF001

        # 2. 不加 emo 且 emotion.enabled=False：强制转语音，但不注入情绪标签
        raw_plain = "/fishaudio 嘿嘿，快看这里！"
        m_plain = cmd_re.search(raw_plain)
        assert m_plain is not None
        ok1, _, intercept1 = await plugin.handle_fishaudio_command(
            text=raw_plain,
            matched_groups=m_plain.groupdict(),
            stream_id="cmd_s1",
        )
        self.assertTrue(ok1)
        self.assertTrue(intercept1)
        self.assertEqual(sent_tts[-1], "嘿嘿，快看这里！")

        # 3. 不加 emo 且 emotion.enabled=True：按设置项自动注入情绪标签并转语音
        plugin.config.emotion.enabled = True
        ok2, _, intercept2 = await plugin.handle_fishaudio_command(
            text=raw_plain,
            matched_groups=m_plain.groupdict(),
            stream_id="cmd_s1",
        )
        self.assertTrue(ok2)
        self.assertTrue(intercept2)
        self.assertEqual(sent_tts[-1], "[开心的语气]嘿嘿，快看这里！")

        # 4. 加 emo 参数且 emotion.enabled=False：强制触发情绪注入并转语音
        plugin.config.emotion.enabled = False
        raw_emo = f"/fishaudio emo {self.SAMPLE_PENGUIN_TEXT}"
        m_emo = cmd_re.search(raw_emo)
        assert m_emo is not None
        ok3, _, intercept3 = await plugin.handle_fishaudio_command(
            text=raw_emo,
            matched_groups=m_emo.groupdict(),
            stream_id="cmd_s1",
        )
        self.assertTrue(ok3)
        self.assertTrue(intercept3)
        joined_emo = "".join(sent_tts[2:])
        self.assertIn("[开心的语气]", joined_emo)
        self.assertIn("[平静]", joined_emo)
        self.assertIn("[语气激动]", joined_emo)
        self.assertIn("[哼歌]", joined_emo)

        # 5. "emotion..." 单词不应误切 emo 前缀
        text_only, force_flag = plugin._parse_fishaudio_command_args({"text": "/fishaudio emotion测试"})  # noqa: SLF001
        self.assertEqual(text_only, "emotion测试")
        self.assertFalse(force_flag)


if __name__ == "__main__":
    unittest.main(verbosity=2)
