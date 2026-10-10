import io
import wave
from array import array
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from price_alert import sounds
from price_alert.launch.alerts import LaunchAlert
from price_alert.launch.rule import LaunchMetrics
from price_alert.models import PriceAlert
from price_alert.streak import StreakAlert
from price_alert.trend.alerts import TrendAlert
from price_alert.trend.pattern import PatternMetrics

NOW = datetime(2026, 1, 1, tzinfo=UTC)
PRICE_ALERT = PriceAlert(
    symbol="BTC_USDT",
    direction="surge",
    price=101.0,
    reference_price=100.0,
    change_percent=1.0,
    move_atr=1.25,
    atr=0.8,
    atr_period=14,
    lookback_seconds=30,
    trade_count=10,
    volume_24h_quote=1_000_000_000,
    timestamp=NOW,
)
STREAK_ALERT = StreakAlert("BTC_USDT", "drop", 3, NOW, 100.0, 97.0, -3.0, 30, 1_000_000_000, NOW)
TREND_ALERT = TrendAlert(
    symbol="ARK_USDT",
    direction="drop",
    period="5m",
    candles=8,
    timestamp=NOW,
    started_at=NOW - timedelta(minutes=40),
    metrics=PatternMetrics("drop", 1.0, 0.92, -8.0, 1, 0.08, 0.58, 0.31, ()),
    volume_24h_quote=50_000_000,
)
LAUNCH_ALERT = LaunchAlert(
    symbol="RLC_USDT",
    timestamp=NOW,
    started_at=NOW - timedelta(minutes=5),
    window_minutes=5,
    baseline_minutes=360,
    metrics=LaunchMetrics("surge", 1.0, 1.05, 5.0, 200_000.0, 10_000.0, 20.0, 1.02, ()),
    volume_24h_quote=1_000_000,
)


def decode(name: sounds.SoundName) -> tuple[wave._wave_params, array]:
    with wave.open(io.BytesIO(sounds.synthesize(name)), "rb") as reader:
        params = reader.getparams()
        samples = array("h", reader.readframes(params.nframes))
    return params, samples


def test_each_alert_kind_maps_to_its_own_sound():
    assert sounds.sound_for(PRICE_ALERT) == "short"
    assert sounds.sound_for(replace(PRICE_ALERT, window="long", lookback_seconds=180)) == "long"
    assert sounds.sound_for(STREAK_ALERT) == "streak"
    assert sounds.sound_for(TREND_ALERT) == "trend"
    assert sounds.sound_for(replace(TREND_ALERT, continuing=True)) == "trend_continuing"
    assert sounds.sound_for(LAUNCH_ALERT) == "launch"
    # 涨跌方向不区分音效。
    assert sounds.sound_for(replace(PRICE_ALERT, direction="drop", change_percent=-1.0)) == "short"


def test_priority_order_is_launch_streak_trend_then_second_level():
    priority = sounds.SOUND_PRIORITY
    assert priority["launch"] > priority["streak"] > priority["trend"] > priority["short"]
    assert priority["trend"] == priority["trend_continuing"]
    assert priority["short"] == priority["long"]
    assert set(priority) == set(sounds.SOUND_NAMES) == set(sounds.SOUND_LABELS)


def test_sounds_are_short_audible_mono_wav_without_clipping():
    for name in sounds.SOUND_NAMES:
        params, samples = decode(name)
        assert (params.nchannels, params.sampwidth, params.framerate) == (1, 2, sounds.SAMPLE_RATE)
        assert params.nframes / params.framerate == sounds.sound_duration(name) <= 1.0
        peak = max(abs(value) for value in samples)
        assert peak == round(sounds.SOUND_SPECS[name].peak_level * 32767) <= 32767
        # 首尾都从静音开始/结束，播放和被打断时没有“咔哒”声。
        assert abs(samples[0]) < 50 and abs(samples[-1]) < 50


def test_launch_sound_is_the_loudest():
    # 放量启动最重要，不能被其他提醒的声音盖过。
    launch = sounds.SOUND_SPECS["launch"]
    others = [spec for name, spec in sounds.SOUND_SPECS.items() if name != "launch"]
    assert all(launch.peak_level > spec.peak_level for spec in others)


def test_frequent_second_level_sounds_are_quieter_than_the_rest():
    # 秒级异动最常响：音量低于趋势与放量，连续提醒稍响。
    specs = sounds.SOUND_SPECS
    for name in ("short", "long", "streak"):
        assert specs[name].peak_level < specs["trend"].peak_level == specs["trend_continuing"].peak_level
    assert specs["short"].peak_level == specs["long"].peak_level < specs["streak"].peak_level


def test_trend_bell_has_no_harsh_high_partials():
    # 钟音的最高分音控制在 4kHz 以内：更高的分音听起来刺耳。
    for name in ("trend", "trend_continuing"):
        spec = sounds.SOUND_SPECS[name]
        highest = max(strike.frequency for strike in spec.strikes) * max(p.ratio for p in spec.timbre)
        assert highest <= 4000


def test_every_sound_is_distinct_and_synthesis_is_deterministic():
    data = [sounds.synthesize(name) for name in sounds.SOUND_NAMES]
    assert len(set(data)) == len(data)
    sounds.synthesize.cache_clear()
    assert [sounds.synthesize(name) for name in sounds.SOUND_NAMES] == data


def test_families_share_timbre_so_each_kind_is_recognizable():
    specs = sounds.SOUND_SPECS
    assert specs["short"].timbre == specs["long"].timbre == specs["streak"].timbre
    assert specs["trend"].timbre == specs["trend_continuing"].timbre
    timbres = {specs["short"].timbre, specs["trend"].timbre, specs["launch"].timbre}
    assert len(timbres) == 3
    # 「敲两下」在两类里都表示连续/延续。
    assert len(specs["streak"].strikes) == len(specs["trend_continuing"].strikes) == 2
    assert len(specs["short"].strikes) == len(specs["trend"].strikes) == 1


def test_sound_files_are_written_once_and_named_by_content(tmp_path):
    paths = sounds.write_sound_files(tmp_path / "sounds")

    assert set(paths) == set(sounds.SOUND_NAMES)
    for name, path in paths.items():
        assert path.read_bytes() == sounds.synthesize(name)
        assert path.name.startswith(f"{name}-") and path.suffix == ".wav"
    mtimes = {path: path.stat().st_mtime_ns for path in paths.values()}

    # 已存在的文件直接复用，不重写（Windows 上正被播放的文件也无法覆盖）。
    assert sounds.write_sound_files(tmp_path / "sounds") == paths
    assert {path: path.stat().st_mtime_ns for path in paths.values()} == mtimes
    assert sorted(p.name for p in (tmp_path / "sounds").iterdir()) == sorted(p.name for p in paths.values())
