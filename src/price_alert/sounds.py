"""控制台提醒音效：按提醒类型用代码合成，Windows 与 macOS 播放同一份 WAV。

音色区分大类、敲击方式区分小类，听到就能分辨是哪一类提醒：

- 秒级异动（短窗口、长窗口、连续提醒）：木琴音色，短促。短窗口敲一下，长窗口音更低、余音更长，
  连续提醒是短窗口的音连敲两下。数量最多，音量最小。
- 趋势（新成立、延续）：清亮但不刺耳的钟音。新成立敲一下，延续连敲两下（与连续提醒一样，“两下”表示延续）。
- 放量启动：明亮的号角音色快速上行琶音（大三和弦），音量最大，最醒目，但收得干脆。
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import tempfile
import wave
from array import array
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from price_alert.detection import Alert
from price_alert.launch.alerts import LaunchAlert
from price_alert.streak import StreakAlert
from price_alert.trend.alerts import TrendAlert

SoundName = Literal["short", "long", "streak", "trend", "trend_continuing", "launch"]
SOUND_NAMES: tuple[SoundName, ...] = ("short", "long", "streak", "trend", "trend_continuing", "launch")
SOUND_LABELS: dict[SoundName, str] = {
    "short": "秒级异动·短窗口",
    "long": "秒级异动·长窗口",
    "streak": "连续提醒",
    "trend": "趋势提醒",
    "trend_continuing": "趋势提醒·延续",
    "launch": "放量启动",
}
# 正在播放时，只有优先级更高的提醒才打断它；同级或更低的跳过，一波异动不会响个不停。
SOUND_PRIORITY: dict[SoundName, int] = {
    "short": 0,
    "long": 0,
    "trend": 1,
    "trend_continuing": 1,
    "streak": 2,
    "launch": 3,
}
SAMPLE_RATE = 44_100
# 峰值留出余量，避免叠加的分音削顶失真。
PEAK_LEVEL = 0.7
# 起音与收尾淡入淡出，避免波形突变产生“咔哒”声。
ATTACK_SECONDS = 0.003
FADE_OUT_SECONDS = 0.02


@dataclass(frozen=True)
class _Partial:
    ratio: float  # 相对基频的倍数；非整数倍听起来像钟、木头这类敲击体
    amplitude: float
    decay_seconds: float  # 指数衰减的时间常数


@dataclass(frozen=True)
class _Strike:
    frequency: float
    start_seconds: float = 0.0
    sustain: float = 1.0  # 衰减时间常数的倍数，>1 余音更长


@dataclass(frozen=True)
class _SoundSpec:
    timbre: tuple[_Partial, ...]
    strikes: tuple[_Strike, ...]
    length_seconds: float
    peak_level: float = PEAK_LEVEL
    attack_seconds: float = ATTACK_SECONDS


# 木琴：第二分音约 3.9 倍、衰减很快，短促不拖尾，适合最常响的秒级异动。
WOOD = (_Partial(1.0, 1.0, 0.16), _Partial(3.93, 0.30, 0.035), _Partial(9.2, 0.08, 0.012))
# 秒级异动的音量：最常响，压低到其他提醒以下；连续提醒稍响一点。
SECOND_LEVEL_PEAK = 0.5
STREAK_PEAK = 0.6
# 钟音：八度泛音和 2.76 倍分音给出清亮的钟声质感，但不要更高的分音——
# 最高分音控制在 4kHz 以内，最初的玻璃音正是 5~10kHz 的分音刺耳；只留基音又显得发闷。
SOFT_BELL = (_Partial(1.0, 1.0, 0.30), _Partial(2.0, 0.30, 0.12), _Partial(2.76, 0.15, 0.06))
# 钟音的起音：比敲击默认的 3 毫秒稍缓，削掉“叮”的冲击，又保留敲击感。
SOFT_ATTACK_SECONDS = 0.006
# 号角：整数倍谐波丰富，明亮，与两种敲击音色差别最大，用于最重要的放量启动。
# 衰减不能太慢：余音拖长会让人觉得啰嗦，醒目靠音色和音量，不靠时长。
FANFARE = (
    _Partial(1.0, 1.0, 0.18),
    _Partial(2.0, 0.6, 0.14),
    _Partial(3.0, 0.4, 0.11),
    _Partial(4.0, 0.25, 0.08),
    _Partial(5.0, 0.15, 0.06),
)
# 琶音相邻两音的间隔：短到连成一个上扬的声音，不会听成“连敲几下”。
ARPEGGIO_GAP = 0.035
# 连敲两下的间隔：足够分辨出两下，又短到仍像“一声”。
DOUBLE_STRIKE_GAP = 0.11

SOUND_SPECS: dict[SoundName, _SoundSpec] = {
    "short": _SoundSpec(WOOD, (_Strike(659.0),), 0.5, SECOND_LEVEL_PEAK),
    # 长窗口音更低、余音更长：窗口越长音越沉，同一音色仍能一听就归到秒级异动。
    "long": _SoundSpec(WOOD, (_Strike(440.0, sustain=1.5),), 0.6, SECOND_LEVEL_PEAK),
    "streak": _SoundSpec(WOOD, (_Strike(659.0), _Strike(659.0, DOUBLE_STRIKE_GAP)), 0.6, STREAK_PEAK),
    "trend": _SoundSpec(SOFT_BELL, (_Strike(1175.0),), 0.6, attack_seconds=SOFT_ATTACK_SECONDS),
    "trend_continuing": _SoundSpec(
        SOFT_BELL,
        (_Strike(1175.0), _Strike(1175.0, DOUBLE_STRIKE_GAP)),
        0.7,
        attack_seconds=SOFT_ATTACK_SECONDS,
    ),
    # C6-E6-G6-C7 上行琶音表示“启动”，落在人耳最敏感的 1~2kHz；顶音余音稍长、整体音量最大，
    # 放量启动最重要，要在其他提醒声中一下就听出来。
    "launch": _SoundSpec(
        FANFARE,
        (
            _Strike(1047.0),
            _Strike(1319.0, ARPEGGIO_GAP),
            _Strike(1568.0, ARPEGGIO_GAP * 2),
            _Strike(2093.0, ARPEGGIO_GAP * 3, sustain=1.2),
        ),
        0.55,
        peak_level=0.95,
    ),
}


def sound_for(alert: Alert) -> SoundName:
    if isinstance(alert, LaunchAlert):
        return "launch"
    if isinstance(alert, StreakAlert):
        return "streak"
    if isinstance(alert, TrendAlert):
        return "trend_continuing" if alert.continuing else "trend"
    return "long" if alert.window == "long" else "short"


def sound_duration(name: SoundName) -> float:
    return SOUND_SPECS[name].length_seconds


@cache
def synthesize(name: SoundName) -> bytes:
    """返回 16 位单声道 WAV 文件内容；纯函数，结果缓存。"""
    spec = SOUND_SPECS[name]
    total = round(spec.length_seconds * SAMPLE_RATE)
    samples = [0.0] * total
    for strike in spec.strikes:
        _add_strike(samples, spec.timbre, strike, spec.attack_seconds)
    peak = max(abs(value) for value in samples) or 1.0
    fade_samples = round(FADE_OUT_SECONDS * SAMPLE_RATE)
    pcm = array("h")
    for index, value in enumerate(samples):
        level = value / peak * spec.peak_level
        remaining = total - index
        if remaining < fade_samples:
            level *= remaining / fade_samples
        pcm.append(round(level * 32767))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        # WAV 规定小端序；array 按本机字节序存储，大端机器需要先翻转。
        if array("h", [1]).tobytes()[0] == 0:
            pcm.byteswap()
        output.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _add_strike(
    samples: list[float], timbre: tuple[_Partial, ...], strike: _Strike, attack_seconds: float
) -> None:
    offset = round(strike.start_seconds * SAMPLE_RATE)
    attack_samples = max(1, round(attack_seconds * SAMPLE_RATE))
    for partial in timbre:
        decay = partial.decay_seconds * strike.sustain
        omega = 2 * math.pi * strike.frequency * partial.ratio / SAMPLE_RATE
        for index in range(len(samples) - offset):
            envelope = partial.amplitude * math.exp(-index / SAMPLE_RATE / decay) * min(index / attack_samples, 1.0)
            samples[offset + index] += envelope * math.sin(omega * index)


def default_sound_dir() -> Path:
    return Path(tempfile.gettempdir()) / "price-alert-sounds"


def write_sound_files(directory: Path) -> dict[SoundName, Path]:
    """把全部音效写成 WAV 文件：afplay 与 winsound 的异步播放都只接受文件路径。

    文件名带内容哈希，调整音效参数后自然换成新文件，不会播到旧的缓存；已存在则直接复用。
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[SoundName, Path] = {}
    for name in SOUND_NAMES:
        data = synthesize(name)
        path = directory / f"{name}-{hashlib.sha256(data).hexdigest()[:12]}.wav"
        if not path.exists():
            # 先写临时文件再改名：多个进程同时启动时，另一个进程不会读到写了一半的文件。
            temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            temporary.write_bytes(data)
            try:
                temporary.replace(path)
            except PermissionError:
                # Windows 上目标文件正被另一个进程播放时无法覆盖；内容相同，用已有的即可。
                temporary.unlink(missing_ok=True)
                if not path.exists():
                    raise
        paths[name] = path
    return paths
