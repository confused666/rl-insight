# Copyright (c) 2026 verl-project authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the NVIDIA and Ascend vendor adapters (all device
tooling is faked: no NVML driver or ``npu-smi`` binary is required)."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

import pytest

from rl_insight.hardware.base import COLLECTOR_REGISTRY
from rl_insight.hardware.vendor.ascend import (
    AscendNpuCollector,
    parse_npu_smi_mapping,
    parse_npu_smi_version,
)
from rl_insight.hardware.vendor.nvidia import NvidiaGpuCollector

_GIB = 1024**3
_MB = 1024 * 1024

# ---------------------------------------------------------------------------
# NVIDIA NVML adapter.
# ---------------------------------------------------------------------------


class _FakeRates:
    """Stand-in for ``nvmlDeviceGetUtilizationRates`` return value."""

    def __init__(self, gpu: int) -> None:
        self.gpu = gpu


class _FakeMemory:
    """Stand-in for ``nvmlDeviceGetMemoryInfo`` return value."""

    def __init__(self, used: int, total: int) -> None:
        self.used = used
        self.total = total


class FakePynvml:
    """pynvml double exposing two uniform devices."""

    NVML_TEMPERATURE_GPU = 0

    def __init__(self, device_count: int = 2) -> None:
        self._device_count = device_count
        self.init_calls = 0

    def nvmlInit(self) -> None:
        self.init_calls += 1

    def nvmlShutdown(self) -> None:
        pass

    def nvmlDeviceGetCount(self) -> int:
        return self._device_count

    def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
        return f"handle-{index}"

    def nvmlDeviceGetName(self, _handle: str) -> bytes | str:
        return b"NVIDIA A100"  # bytes, like older pynvml releases

    def nvmlSystemGetDriverVersion(self) -> str:
        return "550.54.15"

    def nvmlDeviceGetUtilizationRates(self, _handle: str) -> _FakeRates:
        return _FakeRates(42)

    def nvmlDeviceGetMemoryInfo(self, _handle: str) -> _FakeMemory:
        return _FakeMemory(10 * _GIB, 40 * _GIB)

    def nvmlDeviceGetTemperature(self, _handle: str, _sensor: int) -> int:
        return 65

    def nvmlDeviceGetPowerUsage(self, _handle: str) -> int:
        return 250_000  # milliwatts


def test_gpu_collector_should_register_under_config_name() -> None:
    assert COLLECTOR_REGISTRY["gpu"] is NvidiaGpuCollector


def test_gpu_discover_should_follow_capability_and_device_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pynvml = FakePynvml(device_count=2)
    assert NvidiaGpuCollector(pynvml_module=pynvml).discover() is True
    assert pynvml.init_calls == 1

    # A None entry makes ``import pynvml`` raise ImportError deterministically.
    monkeypatch.setitem(sys.modules, "pynvml", None)
    assert NvidiaGpuCollector().discover() is False

    class BrokenInit(FakePynvml):
        def nvmlInit(self) -> None:
            raise RuntimeError("no driver")

    assert NvidiaGpuCollector(pynvml_module=BrokenInit()).discover() is False
    assert (
        NvidiaGpuCollector(pynvml_module=FakePynvml(device_count=0)).discover() is False
    )


def test_gpu_collect_should_report_all_fields_per_device() -> None:
    collector = NvidiaGpuCollector(pynvml_module=FakePynvml(device_count=2))

    samples = collector.collect()

    by_name: dict[str, list[Any]] = {}
    for sample in samples:
        by_name.setdefault(sample.name, []).append(sample)
    assert len(samples) == 14  # 7 metrics x 2 devices
    assert sorted(
        sample.labels["device_index"]
        for sample in by_name["rlinsight_gpu_utilization_percent"]
    ) == [0, 1]
    assert by_name["rlinsight_gpu_utilization_percent"][0].value == 42.0
    assert by_name["rlinsight_gpu_memory_used_bytes"][0].value == 10 * _GIB
    assert by_name["rlinsight_gpu_memory_total_bytes"][0].value == 40 * _GIB
    assert by_name["rlinsight_gpu_memory_utilization_percent"][
        0
    ].value == pytest.approx(25.0)
    assert by_name["rlinsight_gpu_temperature_celsius"][0].value == 65.0
    assert by_name["rlinsight_gpu_power_watts"][0].value == pytest.approx(250.0)

    info = by_name["rlinsight_gpu_info"][0]
    assert info.value == 1.0
    assert info.labels["device_vendor"] == "nvidia"
    assert info.labels["device_name"] == "NVIDIA A100"
    assert info.labels["driver_version"] == "550.54.15"


def test_gpu_collect_should_decode_bytes_and_accept_str_names() -> None:
    class StrNamePynvml(FakePynvml):
        def nvmlDeviceGetName(self, _handle: str) -> str:
            return "H100"  # str, like newer pynvml releases

    samples = NvidiaGpuCollector(pynvml_module=StrNamePynvml()).collect()

    info = next(sample for sample in samples if sample.name == "rlinsight_gpu_info")
    assert info.labels["device_name"] == "H100"


def test_gpu_collect_should_omit_unsupported_fields_but_keep_the_rest() -> None:
    class NoTemperaturePynvml(FakePynvml):
        def nvmlDeviceGetTemperature(self, _handle: str, _sensor: int) -> int:
            raise RuntimeError("not supported")

    samples = NvidiaGpuCollector(pynvml_module=NoTemperaturePynvml()).collect()

    names = {sample.name for sample in samples}
    assert "rlinsight_gpu_temperature_celsius" not in names
    assert "rlinsight_gpu_utilization_percent" in names
    assert len(samples) == 12  # 6 metrics x 2 devices


def test_gpu_collect_should_skip_one_unhealthy_device_but_keep_others() -> None:
    class OneBrokenHandlePynvml(FakePynvml):
        def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
            if index == 1:
                raise RuntimeError("device lost")
            return f"handle-{index}"

    samples = NvidiaGpuCollector(pynvml_module=OneBrokenHandlePynvml()).collect()

    indexes = {sample.labels["device_index"] for sample in samples}
    assert indexes == {0}
    assert len(samples) == 7


def test_gpu_info_should_keep_a_fixed_label_dimension() -> None:
    # One device resolves no name while the other does: both info samples must
    # expose the same label keys (empty value for the unknown one) so the
    # family keeps a single label dimension.
    class OneUnnamedPynvml(FakePynvml):
        def nvmlDeviceGetName(self, handle: str) -> str:
            if handle == "handle-1":
                raise RuntimeError("name unavailable")
            return "NVIDIA A100"

    samples = NvidiaGpuCollector(pynvml_module=OneUnnamedPynvml()).collect()

    info = [sample for sample in samples if sample.name == "rlinsight_gpu_info"]
    assert len(info) == 2
    assert len({tuple(sorted(sample.labels)) for sample in info}) == 1
    by_index = {sample.labels["device_index"]: sample for sample in info}
    assert by_index[0].labels["device_name"] == "NVIDIA A100"
    assert by_index[1].labels["device_name"] == ""


def test_gpu_collect_should_scope_to_integer_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``CUDA_VISIBLE_DEVICES`` uses plain indexes here, which map onto NVML
    # indexes, so a job pinned to a subset only reports those cards.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1, 2")
    collector = NvidiaGpuCollector(pynvml_module=FakePynvml(device_count=4))

    samples = collector.collect()

    assert {sample.labels["device_index"] for sample in samples} == {1, 2}


def test_gpu_collect_should_not_scope_on_non_integer_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # UUID / MIG / PCI-id forms cannot be mapped to NVML indexes; report every
    # device instead of silently dropping all of them.
    for value in ("GPU-abc123", "MIG-abc", "0,GPU-xyz"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        collector = NvidiaGpuCollector(pynvml_module=FakePynvml(device_count=2))

        samples = collector.collect()

        assert {sample.labels["device_index"] for sample in samples} == {0, 1}


# ---------------------------------------------------------------------------
# Ascend npu-smi adapter.
# ---------------------------------------------------------------------------

_MAPPING = """
NPU ID  Chip ID  Chip Logic ID  Chip Name
0       0        0              910B3
1       0        1              910B3
2       0        2              910B3
3       0        3              910B3
"""

_PHYSICAL_MAPPING = """
NPU ID  Chip ID  Chip Logic ID  Chip Phy-ID  Chip Name
0       0        0              0            Ascend950
0       1        1              1            Ascend950
0       2        -              -            Mcu
1       0        2              2            Ascend950
1       1        3              3            Ascend950
1       2        -              -            Mcu
"""


class FakeNpuSmi:
    """``npu-smi`` double serving the mapping and targeted metric queries."""

    def __init__(
        self,
        mapping: str = _MAPPING,
        failures: set[tuple[str, str]] | None = None,
        health_state: str = "OK",
    ) -> None:
        self._mapping = mapping
        self._failures = failures or set()
        self._health_state = health_state
        self.calls: list[tuple[list[str], float]] = []

    def __call__(self, _binary: str, args: Sequence[str], timeout: float) -> str | None:
        arguments = list(args)
        self.calls.append((arguments, timeout))
        if arguments == ["-v"]:
            return "npu-smi Version : 26.1.1"
        if arguments == ["info", "-m"]:
            return self._mapping
        if len(arguments) >= 4 and arguments[:2] == ["info", "-t"]:
            info_type = arguments[2]
            npu_id = arguments[arguments.index("-i") + 1]
            if (info_type, npu_id) in self._failures:
                return None
            if info_type == "usages":
                return (
                    f"NPU ID : {npu_id}\n"
                    "Aicore Usage Rate(%) : 42\n"
                    "HBM Usage Rate(%) : 25\n"
                    "HBM Capacity(MB) : 65536\n"
                    "HBM Bandwidth Usage Rate(%) : 60\n"
                )
            if info_type == "temp":
                return f"NPU ID : {npu_id}\nChip ID : 0\nTemperature (C) : 49"
            if info_type == "power":
                return f"NPU ID : {npu_id}\nChip ID : 0\nPower (W) : 97.2"
            if info_type == "health":
                return (
                    f"NPU ID : {npu_id}\nChip ID : 0\n"
                    f"Health Status : {self._health_state}"
                )
        return None


def _by_name_and_index(samples: list[Any]) -> dict[tuple[str, str], Any]:
    """Index samples by (metric name, device index) for convenient lookup."""
    return {
        (sample.name, str(sample.labels["device_index"])): sample for sample in samples
    }


def test_npu_collector_should_register_under_config_name() -> None:
    assert COLLECTOR_REGISTRY["npu"] is AscendNpuCollector


def test_parse_mapping_should_read_compute_chips() -> None:
    chips = parse_npu_smi_mapping(_MAPPING)

    assert [(chip.npu_id, chip.chip_id, chip.logic_id) for chip in chips] == [
        ("0", "0", "0"),
        ("1", "0", "1"),
        ("2", "0", "2"),
        ("3", "0", "3"),
    ]
    assert all(chip.name == "910B3" for chip in chips)
    assert all(chip.chip_query for chip in chips)


def test_parse_mapping_should_skip_mcu_rows_and_omit_chip_query() -> None:
    # Ascend 950 reports physical chip ids and expects ``-i <npu>`` without
    # ``-c``; MCU rows carry ``-`` as their logic id and are not compute chips.
    chips = parse_npu_smi_mapping(_PHYSICAL_MAPPING)

    assert [(chip.npu_id, chip.chip_id, chip.logic_id) for chip in chips] == [
        ("0", "0", "0"),
        ("0", "1", "1"),
        ("1", "0", "2"),
        ("1", "1", "3"),
    ]
    assert all(not chip.chip_query for chip in chips)


def test_parse_mapping_should_return_empty_for_malformed_output() -> None:
    assert parse_npu_smi_mapping("") == []
    assert parse_npu_smi_mapping("totally unrelated text\n") == []


def test_parse_version_should_read_the_last_colon_field() -> None:
    assert parse_npu_smi_version("npu-smi Version : 26.1.1\n") == "26.1.1"


def test_npu_discover_should_follow_binary_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rl_insight.hardware.vendor.ascend.shutil.which", lambda _name: None
    )
    assert AscendNpuCollector(runner=FakeNpuSmi()).discover() is False

    monkeypatch.setattr(
        "rl_insight.hardware.vendor.ascend.shutil.which",
        lambda _name: "/usr/local/sbin/npu-smi",
    )
    assert AscendNpuCollector(runner=FakeNpuSmi()).discover() is True


def test_npu_discover_should_cache_the_mapping_and_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rl_insight.hardware.vendor.ascend.shutil.which",
        lambda _name: "/usr/local/sbin/npu-smi",
    )
    runner = FakeNpuSmi()
    collector = AscendNpuCollector(runner=runner)

    assert collector.discover() is True
    assert [args for args, _ in runner.calls] == [["info", "-m"], ["-v"]]

    runner.calls.clear()
    collector.collect()
    assert ["info", "-m"] not in [args for args, _ in runner.calls]


def test_npu_collect_should_use_targeted_subcommands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "rl_insight.hardware.vendor.ascend.shutil.which",
        lambda _name: "/usr/local/sbin/npu-smi",
    )
    runner = FakeNpuSmi()
    collector = AscendNpuCollector(runner=runner)
    collector.discover()
    runner.calls.clear()

    samples = collector.collect()

    assert len(samples) == 36  # 9 metrics x 4 chips
    indexed = _by_name_and_index(samples)
    device_0 = {name: indexed[(name, "0")] for name, index in indexed if index == "0"}
    assert device_0["rlinsight_npu_utilization_percent"].value == 42.0
    assert device_0["rlinsight_npu_power_watts"].value == 97.2
    assert device_0["rlinsight_npu_temperature_celsius"].value == 49.0
    assert device_0["rlinsight_npu_health_status"].value == 1.0
    assert device_0["rlinsight_npu_hbm_bandwidth_utilization_percent"].value == 60.0
    assert device_0["rlinsight_npu_memory_total_bytes"].value == 65536 * _MB
    assert device_0["rlinsight_npu_memory_used_bytes"].value == 25 / 100 * 65536 * _MB
    assert device_0["rlinsight_npu_memory_utilization_percent"].value == 25.0

    info = device_0["rlinsight_npu_info"]
    assert info.value == 1.0
    assert info.labels["device_vendor"] == "ascend"
    assert info.labels["chip_name"] == "910B3"
    assert info.labels["npu_smi_version"] == "26.1.1"

    expected_args = [
        ["info", "-t", info_type, "-i", npu_id, "-c", "0"]
        for npu_id in ("0", "1", "2", "3")
        for info_type in ("usages", "temp", "power", "health")
    ]
    assert [args for args, _ in runner.calls] == expected_args
    assert all(0.0 < timeout <= 2.0 for _, timeout in runner.calls)


def test_npu_collect_should_omit_chip_flag_for_physical_ids() -> None:
    runner = FakeNpuSmi(mapping=_PHYSICAL_MAPPING)

    samples = AscendNpuCollector(runner=runner).collect()

    assert len(samples) == 36  # 9 metrics x 4 compute chips
    assert all("-c" not in args for args, _ in runner.calls)


def test_npu_health_should_map_non_ok_states_to_unhealthy() -> None:
    collector = AscendNpuCollector(runner=FakeNpuSmi(health_state="Warning"))

    health = [
        sample
        for sample in collector.collect()
        if sample.name == "rlinsight_npu_health_status"
    ]

    assert len(health) == 4
    assert all(sample.value == 0.0 for sample in health)


def test_npu_collect_should_scope_to_the_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``npu-smi`` lists every chip on the node, so a job pinned to a subset
    # must only publish samples for the cards it can actually use.
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2, 3")
    monkeypatch.setattr(
        "rl_insight.hardware.vendor.ascend.shutil.which",
        lambda _name: "/usr/local/sbin/npu-smi",
    )
    runner = FakeNpuSmi()
    collector = AscendNpuCollector(runner=runner)
    collector.discover()
    runner.calls.clear()

    samples = collector.collect()

    assert len(samples) == 18  # 9 metrics x 2 visible chips
    assert {str(sample.labels["device_index"]) for sample in samples} == {"2", "3"}
    assert {args[args.index("-i") + 1] for args, _ in runner.calls} == {"2", "3"}


def test_npu_collect_should_not_scope_on_non_integer_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,GPU-abc")
    collector = AscendNpuCollector(runner=FakeNpuSmi())

    assert len(collector.collect()) == 36


def test_npu_collect_should_omit_failed_fields_but_keep_the_chip() -> None:
    runner = FakeNpuSmi(failures={("usages", "0"), ("temp", "1")})
    collector = AscendNpuCollector(runner=runner)

    samples = collector.collect()

    indexed = _by_name_and_index(samples)
    assert ("rlinsight_npu_utilization_percent", "0") not in indexed
    assert ("rlinsight_npu_memory_used_bytes", "0") not in indexed
    assert ("rlinsight_npu_power_watts", "0") in indexed
    assert ("rlinsight_npu_temperature_celsius", "0") in indexed

    assert ("rlinsight_npu_temperature_celsius", "1") not in indexed
    assert ("rlinsight_npu_power_watts", "1") in indexed


def test_npu_collect_should_return_empty_when_the_mapping_fails() -> None:
    collector = AscendNpuCollector(runner=FakeNpuSmi(mapping=""))

    assert collector.collect() == []


def test_npu_info_should_keep_a_fixed_label_dimension() -> None:
    # Chip 1 reports no product name: both info samples must expose the same
    # label keys (empty value for the unknown name) so the family keeps a
    # single label dimension and the hub registry accepts every chip.
    mapping = """
NPU ID  Chip ID  Chip Logic ID  Chip Name
0       0        0              910B3
1       0        1
"""
    collector = AscendNpuCollector(runner=FakeNpuSmi(mapping=mapping))

    samples = collector.collect()

    info = [sample for sample in samples if sample.name == "rlinsight_npu_info"]
    assert len(info) == 2
    assert len({tuple(sorted(sample.labels)) for sample in info}) == 1
    by_index = {sample.labels["device_index"]: sample for sample in info}
    assert by_index["0"].labels["chip_name"] == "910B3"
    assert by_index["1"].labels["chip_name"] == ""
