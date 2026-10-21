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

"""Ascend NPU collector backed by targeted ``npu-smi`` subcommands."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

from ..base import (
    HardwareCollector,
    Sample,
    optional_sample,
    present_samples,
    register_collector,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = [
    "AscendNpuCollector",
    "parse_npu_smi_mapping",
    "parse_npu_smi_version",
]

_NPU_SMI_BINARY = "npu-smi"
# One targeted query is a small CLI call; keep the per-call timeout short so a
# hung driver cannot stall the sampling thread.
_NPU_SMI_TIMEOUT_SECONDS = 2.0
# Overall budget for the metric queries of one round. With four calls per
# chip this fits roughly six chips at the per-call timeout, and caps the
# worst case on a large node instead of multiplying the timeout by the card
# count.
_NPU_SMI_ROUND_BUDGET_SECONDS = 5.0
_VISIBLE_DEVICES_ENV = "ASCEND_RT_VISIBLE_DEVICES"
_MB_TO_BYTES = 1024 * 1024

# ``npu-smi info -t health`` reports one of OK, Warning, Alarm, Critical, or
# UNKNOWN. Map it to the exporter's binary semantics (1=healthy, 0=unhealthy).
_HEALTHY_STATES = frozenset({"OK", "HEALTHY", "1"})
_UNHEALTHY_STATES = frozenset(
    {"WARNING", "ALARM", "CRITICAL", "UNKNOWN", "UNHEALTHY", "0"}
)

# Runner signature: (binary, arguments, timeout_seconds) -> stdout text, or
# None on failure.
NpuSmiRunner = Callable[[str, Sequence[str], float], "str | None"]


def _run_npu_smi(binary: str, args: Sequence[str], timeout: float) -> str | None:
    """Run one ``npu-smi`` command and return its stdout, or ``None``."""
    try:
        result = subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


@dataclass(frozen=True)
class _Chip:
    """One compute chip from ``npu-smi info -m``."""

    npu_id: str
    chip_id: str
    logic_id: str
    name: str
    # Ascend 950 uses physical chip ids and expects ``-i <npu>`` without
    # ``-c``; older products address a chip with ``-i <npu> -c <chip>``.
    chip_query: bool = True


def parse_npu_smi_mapping(text: str) -> list[_Chip]:
    """Parse ``npu-smi info -m`` into compute chips.

    The command prints either ``NPU ID / Chip ID / Chip Logic ID / Chip
    Name`` or, on Ascend 950, ``NPU ID / Chip ID / Chip Logic ID / Chip
    Phy-ID / Chip Name``. MCU rows carry ``-`` as their logic id and are
    skipped, as are malformed rows, so one bad line never hides the other
    chips.

    Args:
        text: Raw ``npu-smi info -m`` stdout.

    Returns:
        Chips sorted by logical id (the id used by ``ASCEND_RT_VISIBLE_DEVICES``).
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header_index = next(
        (index for index, line in enumerate(lines) if "Chip Logic ID" in line),
        None,
    )
    if header_index is None:
        return []

    physical_ids = "Chip Phy-ID" in lines[header_index]
    chips: list[_Chip] = []
    for line in lines[header_index + 1 :]:
        parts = line.replace("|", " ").split()
        if physical_ids:
            if len(parts) < 4:
                continue
            npu_id, chip_id, logic_id = parts[0], parts[1], parts[2]
            name = " ".join(parts[4:])
        else:
            if len(parts) < 3:
                continue
            npu_id, chip_id, logic_id = parts[0], parts[1], parts[2]
            name = " ".join(parts[3:])
        if not (npu_id.isdigit() and chip_id.isdigit() and logic_id.isdigit()):
            continue
        chips.append(
            _Chip(
                npu_id=npu_id,
                chip_id=chip_id,
                logic_id=logic_id,
                name=name,
                chip_query=not physical_ids,
            )
        )
    return sorted(chips, key=lambda chip: int(chip.logic_id))


def parse_npu_smi_version(text: str) -> str:
    """Extract the version from ``npu-smi -v`` output."""
    return text.strip().split(":")[-1].strip()


def _number_after_colon(line: str) -> float | None:
    """Parse the first number after the last ``:`` on one output line."""
    tokens = line.split(":")[-1].split()
    if not tokens:
        return None
    try:
        return float(tokens[0])
    except ValueError:
        return None


def _field_number(text: str, field: str) -> float | None:
    """Return the value of a ``field : number`` line, or ``None``."""
    lowered = field.lower()
    for line in text.splitlines():
        if lowered in line.lower():
            return _number_after_colon(line)
    return None


def _health_status(text: str) -> float | None:
    """Map an ``npu-smi -t health`` status to 1 (healthy) or 0 (unhealthy)."""
    for line in text.splitlines():
        lowered = line.lower()
        if "health status" in lowered or lowered.strip().startswith("health"):
            state = line.split(":")[-1].strip().upper()
            if state in _HEALTHY_STATES:
                return 1.0
            if state in _UNHEALTHY_STATES:
                return 0.0
    return None


def _visible_device_ids() -> set[str] | None:
    """Logical device ids this process may use, or ``None`` unrestricted.

    ``npu-smi`` reports every chip on the node regardless of the caller's
    environment, so a job pinned to a subset (``ASCEND_RT_VISIBLE_DEVICES``)
    would otherwise publish samples for cards it cannot reach. Non-integer
    values cannot be mapped onto logic ids, so they leave the collector
    unrestricted instead of dropping every chip.
    """
    raw = os.environ.get(_VISIBLE_DEVICES_ENV, "").strip()
    if not raw:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens or not all(token.isdigit() for token in tokens):
        return None
    return set(tokens)


class AscendNpuCollector(HardwareCollector):
    """Per-chip Ascend NPU metrics through targeted ``npu-smi`` queries.

    ``npu-smi`` is a host-side management tool that may be absent inside
    training containers; ``discover`` only checks that the binary is on
    ``PATH`` so missing tooling degrades to an unavailable status instead of
    failing. The static chip mapping is queried once and cached, and each
    round issues four small commands per visible chip (``usages``, ``temp``,
    ``power``, ``health``) bounded by a shared round budget so the sampler
    stays cheap on large nodes.

    Reporting is scoped to ``ASCEND_RT_VISIBLE_DEVICES`` when that variable
    holds integer logic ids, so a job using a subset of the node's cards only
    publishes samples for those cards.
    """

    name = "npu"

    def __init__(
        self,
        binary: str = _NPU_SMI_BINARY,
        runner: NpuSmiRunner | None = None,
    ) -> None:
        """
        Args:
            binary: ``npu-smi`` executable name or path.
            runner: Injected command runner for tests; defaults to bounded
                ``npu-smi`` invocations.
        """
        self._binary = binary
        self._runner: NpuSmiRunner = runner or _run_npu_smi
        self._chips: list[_Chip] = []
        self._version = ""

    def discover(self) -> bool:
        """Report whether ``npu-smi`` is usable, caching the chip mapping.

        Only tooling availability decides the result: a node whose binary
        exists but reports no cards still counts as available, and ``collect``
        then yields no samples. The mapping and version are cached here, so a
        round only pays for the dynamic metric queries.
        """
        if shutil.which(self._binary) is None:
            logger.info(
                "[rl-insight] npu-smi was not found on PATH; Ascend NPU "
                "metrics are unavailable on this node."
            )
            return False
        self._load_chips()
        output = self._runner(self._binary, ["-v"], _NPU_SMI_TIMEOUT_SECONDS)
        if output:
            self._version = parse_npu_smi_version(output)
        return True

    def collect(self) -> list[Sample]:
        chips = self._chips or self._load_chips()
        if not chips:
            return []

        visible = _visible_device_ids()
        deadline = time.monotonic() + _NPU_SMI_ROUND_BUDGET_SECONDS
        samples: list[Sample] = []
        for chip in chips:
            if visible is not None and chip.logic_id not in visible:
                continue
            if time.monotonic() >= deadline:
                logger.warning(
                    "[rl-insight] npu-smi sampling exceeded its %.1fs round "
                    "budget; skipping the remaining chips.",
                    _NPU_SMI_ROUND_BUDGET_SECONDS,
                )
                break
            samples.extend(self._collect_chip(chip, deadline))
        return samples

    def _load_chips(self) -> list[_Chip]:
        """Query and cache the static chip mapping."""
        output = self._runner(self._binary, ["info", "-m"], _NPU_SMI_TIMEOUT_SECONDS)
        chips = parse_npu_smi_mapping(output) if output else []
        if chips:
            self._chips = chips
        elif output:
            logger.info("[rl-insight] npu-smi reported no compute chips.")
        return chips

    def _collect_chip(self, chip: _Chip, deadline: float) -> list[Sample]:
        """Collect one chip; each failed query only omits its own metrics."""
        labels = {"device_vendor": "ascend", "device_index": chip.logic_id}

        usage = self._query("usages", chip, deadline)
        utilization = _field_number(usage, "aicore usage rate") if usage else None
        hbm_rate = _field_number(usage, "hbm usage rate") if usage else None
        hbm_capacity_mb = _field_number(usage, "hbm capacity") if usage else None
        hbm_bandwidth = (
            _field_number(usage, "hbm bandwidth usage rate") if usage else None
        )
        hbm_total = None if hbm_capacity_mb is None else hbm_capacity_mb * _MB_TO_BYTES
        hbm_used = (
            None
            if hbm_rate is None or hbm_total is None
            else hbm_rate / 100.0 * hbm_total
        )

        temperature = _field_number(
            self._query("temp", chip, deadline) or "", "temperature"
        )
        power = _field_number(self._query("power", chip, deadline) or "", "power")
        health = _health_status(self._query("health", chip, deadline) or "")

        samples = present_samples(
            [
                optional_sample(
                    "rlinsight_npu_utilization_percent",
                    utilization,
                    labels,
                    "Ascend NPU AICore utilization percent",
                ),
                optional_sample(
                    "rlinsight_npu_power_watts",
                    power,
                    labels,
                    "Ascend NPU power draw in watts",
                ),
                optional_sample(
                    "rlinsight_npu_temperature_celsius",
                    temperature,
                    labels,
                    "Ascend NPU temperature in celsius",
                ),
                optional_sample(
                    "rlinsight_npu_health_status",
                    health,
                    labels,
                    "Ascend NPU health (1=healthy, 0=unhealthy)",
                ),
                optional_sample(
                    "rlinsight_npu_hbm_bandwidth_utilization_percent",
                    hbm_bandwidth,
                    labels,
                    "Ascend NPU HBM bandwidth utilization percent",
                ),
                optional_sample(
                    "rlinsight_npu_memory_used_bytes",
                    hbm_used,
                    labels,
                    "Ascend NPU HBM used in bytes",
                ),
                optional_sample(
                    "rlinsight_npu_memory_total_bytes",
                    hbm_total,
                    labels,
                    "Ascend NPU HBM total in bytes",
                ),
                optional_sample(
                    "rlinsight_npu_memory_utilization_percent",
                    hbm_rate,
                    labels,
                    "Ascend NPU HBM utilization percent",
                ),
            ]
        )

        # Keep the label dimension fixed across chips: a chip without a
        # product name must still carry the key (with an empty value),
        # otherwise the hub's registry would reject the batch as a duplicate
        # time series.
        info_labels = dict(labels)
        info_labels["chip_name"] = chip.name
        info_labels["npu_smi_version"] = self._version
        samples.append(
            Sample(
                name="rlinsight_npu_info",
                value=1.0,
                labels=info_labels,
                documentation="Ascend NPU identity (always 1; see labels)",
            )
        )
        return samples

    def _query(self, info_type: str, chip: _Chip, deadline: float) -> str | None:
        """Run one targeted query within the remaining round budget."""
        remaining = min(_NPU_SMI_TIMEOUT_SECONDS, deadline - time.monotonic())
        if remaining <= 0:
            return None
        args = ["info", "-t", info_type, "-i", chip.npu_id]
        if chip.chip_query:
            args.extend(["-c", chip.chip_id])
        return self._runner(self._binary, args, remaining)


register_collector(AscendNpuCollector.name, AscendNpuCollector)
