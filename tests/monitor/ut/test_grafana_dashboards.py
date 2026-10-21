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

"""Unit tests for Grafana dashboards: runtime staging and hardware panels."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from rl_insight.server import runtime as runtime_module
from rl_insight.utils.constants import MonitorDefaults

# ---------------------------------------------------------------------------
# Dashboard runtime staging.
# ---------------------------------------------------------------------------


def _conf(builtin: Path, data_dir: Path, extra: Path | None = None):
    grafana = {"dashboards_dir": str(builtin)}
    if extra is not None:
        grafana["extra_dashboard_dir"] = str(extra)
    return OmegaConf.create({"grafana": grafana, "server": {"data_dir": str(data_dir)}})


def test_stage_copies_builtin_and_extra_without_parsing_json(tmp_path) -> None:
    builtin = tmp_path / "builtin"
    extra = tmp_path / "extra"
    (builtin / "verl").mkdir(parents=True)
    (extra / "custom").mkdir(parents=True)
    (builtin / "verl" / "board.json").write_text("{}", encoding="utf-8")
    (extra / "custom" / "router.json").write_text("not valid json", encoding="utf-8")

    staged = runtime_module._stage_grafana_dashboards(
        _conf(builtin, tmp_path / "state", extra), tmp_path / "runtime"
    )

    assert (staged / "verl" / "board.json").is_file()
    assert (staged / "custom" / "router.json").read_text(encoding="utf-8") == (
        "not valid json"
    )


def test_stage_rejects_json_collision_before_copying_extra(tmp_path) -> None:
    builtin = tmp_path / "builtin"
    extra = tmp_path / "extra"
    (builtin / "shared").mkdir(parents=True)
    (extra / "shared").mkdir(parents=True)
    (builtin / "shared" / "board.json").write_text("builtin", encoding="utf-8")
    (extra / "unique.json").write_text("extra", encoding="utf-8")
    (extra / "shared" / "board.json").write_text("conflict", encoding="utf-8")

    runtime_dir = tmp_path / "runtime"
    with pytest.raises(RuntimeError, match="already exists in runtime dashboards"):
        runtime_module._stage_grafana_dashboards(
            _conf(builtin, tmp_path / "state", extra),
            runtime_dir,
        )

    staged = runtime_dir / "dashboards"
    assert (staged / "shared" / "board.json").read_text(encoding="utf-8") == "builtin"
    assert not (staged / "unique.json").exists()


def test_stage_refreshes_extra_in_shared_builtin_directory(tmp_path) -> None:
    builtin = tmp_path / "builtin"
    extra = tmp_path / "extra"
    (builtin / "shared").mkdir(parents=True)
    (extra / "shared").mkdir(parents=True)
    (builtin / "shared" / "builtin.json").write_text("builtin", encoding="utf-8")
    extra_dashboard = extra / "shared" / "extra.json"
    extra_dashboard.write_text("first", encoding="utf-8")
    runtime_dir = tmp_path / "runtime"

    runtime_module._stage_grafana_dashboards(
        _conf(builtin, tmp_path / "state", extra), runtime_dir
    )
    extra_dashboard.write_text("second", encoding="utf-8")
    staged = runtime_module._stage_grafana_dashboards(
        _conf(builtin, tmp_path / "state", extra), runtime_dir
    )

    assert (staged / "shared" / "extra.json").read_text(encoding="utf-8") == "second"


@pytest.mark.parametrize(
    ("entry_type", "expected_error"),
    [("missing", "does not exist"), ("file", "is not a directory")],
)
def test_stage_rejects_invalid_extra_directory(
    tmp_path, entry_type: str, expected_error: str
) -> None:
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    extra = tmp_path / entry_type
    if entry_type == "file":
        extra.write_text("not a directory", encoding="utf-8")

    with pytest.raises(RuntimeError, match=expected_error):
        runtime_module._stage_grafana_dashboards(
            _conf(builtin, tmp_path / "state", extra), tmp_path / "runtime"
        )


# ---------------------------------------------------------------------------
# Trainer dashboard hardware panels.
#
# Every hardware panel reads the in-process metrics emitted by
# ``rl_insight.init()``; the legacy exporter queries were removed so the
# built-in collector is the only hardware source.
# ---------------------------------------------------------------------------

_DASHBOARDS_DIR = (
    Path(__file__).resolve().parents[3]
    / "rl_insight"
    / "config"
    / "services"
    / "grafana"
    / "dashboards"
)
_TRAINER_DASHBOARDS = [
    _DASHBOARDS_DIR / "verl" / "verl_trainer_v1_with_vllm_engine.json",
    _DASHBOARDS_DIR / "verl-omni" / "verl_omni_trainer_v1_with_vllm_omni_engine.json",
]

# Prometheus prefix added by the hub's MetricRegistry namespace.
_METRIC_PREFIX = f"{MonitorDefaults.NAMESPACE}_rlinsight_"

# Hardware panel id -> in-process metric family backing each panel.
_INTEGRATED_PANELS = {
    "325": "cpu_utilization_percent",
    "326": "memory_used_bytes",
    "328": "network_receive_bytes_per_second",
    "329": "npu_utilization_percent",
    "330": "npu_memory_utilization_percent",
    "331": "npu_memory_used_bytes",
    "332": "npu_power_watts",
    "333": "npu_temperature_celsius",
    "334": "npu_health_status",
    "335": "npu_hbm_bandwidth_utilization_percent",
    "337": "gpu_utilization_percent",
    "338": "gpu_memory_utilization_percent",
    "339": "gpu_memory_used_bytes",
    "340": "gpu_power_watts",
    "341": "gpu_temperature_celsius",
}
# In-process device panels: panel ids with NPU legend vs GPU legend.
_NPU_DEVICE_PANELS = ("329", "330", "331", "332", "333", "334", "335")
_GPU_DEVICE_PANELS = ("337", "338", "339", "340", "341")
# Node-level host panels that must say so, since their values cover every
# process on the node rather than just this training job.
_HOST_PANELS = ("325", "326", "328", "344")
# In-process-only process panels: panel id -> backing metric family.
# The per-Ray-worker panels read the dedicated rlinsight_ray_process_* families
# (on by default) and are filtered by card; the self-process panels read
# rlinsight_process_* and have no card to show.
_RAY_PROCESS_PANELS = {
    "345": "ray_process_cpu_percent",
    "346": "ray_process_rss_bytes",
}
_SELF_PROCESS_PANELS = {
    "347": "process_cpu_percent",
    "348": "process_rss_bytes",
}
_RAY_PROCESS_LEGEND = "{{node}} {{device_index}} {{worker_role}} {{pid}}"
_SELF_PROCESS_LEGEND = "{{node}} {{worker_role}} {{pid}}"
_DEVICE_FILTER = 'device_index=~"$device_index"'


def _device_filter_selects(expr: str) -> bool:
    """Whether the device filter is a label matcher inside a metric selector.

    Appending the matcher behind the selector's closing brace silently turns it
    into an extra ``label_replace`` argument, which makes PromQL reject the whole
    expression and the panel go blank.
    """
    return (
        re.search(r"\{[^{}]*" + re.escape(_DEVICE_FILTER) + r"[^{}]*\}", expr)
        is not None
    )


# Node scoping behind the node tabs: every panel matches the selected node
# through the in-process ``node`` label.
_NODE_FILTER = 'node=~"$node"'
# Panel ids scoped by the selected node. Both the NPU and the NVIDIA GPU
# device rows follow the node picker.
_NODE_SCOPED_PANELS = (
    "325",
    "326",
    "328",
    *_NPU_DEVICE_PANELS,
    *_GPU_DEVICE_PANELS,
    "344",
    *_RAY_PROCESS_PANELS,
    *_SELF_PROCESS_PANELS,
)
# Every hardware timeseries panel.
_HW_TIMESERIES_PANELS = tuple(str(panel_id) for panel_id in range(325, 349))


def _node_filter_selects(expr: str) -> bool:
    """Whether the node filter is a label matcher inside a metric selector."""
    return (
        re.search(r"\{[^{}]*" + re.escape(_NODE_FILTER) + r"[^{}]*\}", expr) is not None
    )


@pytest.fixture(
    scope="module", params=_TRAINER_DASHBOARDS, ids=lambda path: path.parent.name
)
def trainer_dashboard(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Parsed spec of one trainer dashboard."""
    with open(request.param, encoding="utf-8") as f:
        return json.load(f)["spec"]


def _panel_queries(spec: dict[str, Any], panel_id: str) -> list[tuple[str, str, bool]]:
    """(expr, legendFormat, hidden) triples of one hardware panel."""
    queries = spec["elements"][f"panel-{panel_id}"]["spec"]["data"]["spec"]["queries"]
    return [
        (
            query["spec"]["query"]["spec"]["expr"],
            query["spec"]["query"]["spec"].get("legendFormat", ""),
            bool(query["spec"].get("hidden")),
        )
        for query in queries
    ]


def _visible(spec: dict[str, Any], panel_id: str) -> list[str]:
    return [expr for expr, _, hidden in _panel_queries(spec, panel_id) if not hidden]


def _hidden(spec: dict[str, Any], panel_id: str) -> list[str]:
    return [expr for expr, _, hidden in _panel_queries(spec, panel_id) if hidden]


def test_integrated_panels_should_show_in_process_queries(
    trainer_dashboard: dict[str, Any],
) -> None:
    for panel_id, family in _INTEGRATED_PANELS.items():
        visible = _visible(trainer_dashboard, panel_id)
        assert visible, panel_id
        assert _hidden(trainer_dashboard, panel_id) == [], panel_id
        for expr in visible:
            assert _METRIC_PREFIX in expr, expr
            assert 'project=~"$project"' in expr, expr
            assert 'experiment_name=~"$experiment_name"' in expr, expr
        assert any(f"{_METRIC_PREFIX}{family}" in expr for expr in visible), panel_id

    # The network panel reports receive and transmit as separate queries.
    network_visible = _visible(trainer_dashboard, "328")
    assert any("network_receive_bytes_per_second" in expr for expr in network_visible)
    assert any("network_transmit_bytes_per_second" in expr for expr in network_visible)


@pytest.mark.parametrize(
    "dashboard_path", _TRAINER_DASHBOARDS, ids=lambda path: path.parent.name
)
def test_disk_panel_should_show_in_process_queries(dashboard_path: Path) -> None:
    with open(dashboard_path, encoding="utf-8") as f:
        spec = json.load(f)["spec"]

    visible = _visible(spec, "344")
    assert visible
    assert _hidden(spec, "344") == []
    for expr in visible:
        assert _METRIC_PREFIX in expr, expr
        assert 'project=~"$project"' in expr, expr
        assert 'experiment_name=~"$experiment_name"' in expr, expr

    assert any("disk_read_bytes_per_second" in expr for expr in visible)
    assert any("disk_write_bytes_per_second" in expr for expr in visible)


def test_npu_panels_should_keep_id_legend(
    trainer_dashboard: dict[str, Any],
) -> None:
    # The in-process device_index is renamed to id, and node prefixes the
    # legend so multi-node series never collide.
    for panel_id in _NPU_DEVICE_PANELS:
        for expr, legend, hidden in _panel_queries(trainer_dashboard, panel_id):
            assert not hidden, expr
            assert "label_replace(" in expr, expr
            assert legend == "{{node}} NPU {{id}}", legend
            assert '"device_index"' in expr, expr


def test_gpu_panels_should_keep_id_legend(
    trainer_dashboard: dict[str, Any],
) -> None:
    # The in-process series rewrite device_index to id and carry node directly.
    for panel_id in _GPU_DEVICE_PANELS:
        for expr, legend, hidden in _panel_queries(trainer_dashboard, panel_id):
            assert not hidden, expr
            assert "label_replace(" in expr, expr
            assert legend == "{{node}} GPU {{id}}", legend
            assert '"device_index"' in expr, expr


def test_memory_used_panels_should_scale_in_process_bytes_to_megabytes(
    trainer_dashboard: dict[str, Any],
) -> None:
    # The in-process series are bytes; the panels convert them to megabytes.
    for panel_id in ("331", "339"):
        visible = _visible(trainer_dashboard, panel_id)
        assert visible
        for expr in visible:
            assert "/ 1048576" in expr, expr


def test_node_variable_should_read_the_in_process_node_label(
    trainer_dashboard: dict[str, Any],
) -> None:
    # The node picker is sourced only from the in-process series; there is no
    # exporter instance to synthesize a node from anymore.
    spec = next(
        variable["spec"]
        for variable in trainer_dashboard["variables"]
        if variable["spec"]["name"] == "node"
    )
    query = spec["definition"]

    assert "query_result(" in query
    assert "count by (node)" in query
    assert 'node=~".+"' in query
    assert 'node!~".+"' not in query
    assert "label_replace(" not in query
    assert f"{_METRIC_PREFIX}npu_info" in query
    assert f"{_METRIC_PREFIX}cpu_utilization_percent" in query
    assert "npu_chip_info_name" not in query
    # Grafana anchors an undelimited pattern with ``^...$`` (``stringToJsRegex``),
    # which can never match the ``query_result`` text
    # ``{node="1.2.3.4"} 1 1712345678000``; the pattern must be slash delimited,
    # and the capture named ``value`` so it becomes the option value.
    assert spec["regex"] == '/node="(?<value>[^"]+)"/'
    assert spec["regexApplyTo"] == "value"
    assert spec["hide"] == "dontHide"
    assert spec["multi"] is True
    assert spec["includeAll"] is True
    assert spec["allValue"] == ".*"
    assert spec["current"] == {"text": "All", "value": "$__all"}
    assert spec["query"]["spec"]["query"] == query
    assert spec["query"]["datasource"] == {"name": "${datasource}"}


def test_variable_regexes_should_be_slash_delimited(
    trainer_dashboard: dict[str, Any],
) -> None:
    # Grafana wraps an undelimited pattern in ``^...$`` (``stringToJsRegex``),
    # so ``node="([^"]+)"`` silently matched nothing against the query_result
    # text ``{node="1.2.3.4"} 1 1712345678000`` and the variable ended up with
    # zero options. Anything but ``/re/`` gets anchored, so every non-empty
    # regex must be slash delimited.
    for variable in trainer_dashboard["variables"]:
        name = variable["spec"]["name"]
        regex = variable["spec"].get("regex", "")
        if regex:
            assert re.fullmatch(r"/.*/[gmi]*", regex), (name, regex)


def test_worker_role_variable_should_read_both_process_metric_families(
    trainer_dashboard: dict[str, Any],
) -> None:
    # The dropdown must cover both the default per-Ray-worker families and the
    # opt-in self-process ones, so a manually set hardware.worker_role is also
    # selectable.
    variable = next(
        variable
        for variable in trainer_dashboard["variables"]
        if variable["spec"]["name"] == "worker_role"
    )
    spec = variable["spec"]
    query = spec["definition"]

    assert f"{_METRIC_PREFIX}ray_process_cpu_percent" in query
    assert f"{_METRIC_PREFIX}process_cpu_percent" in query
    assert "worker_role" in query
    assert spec["multi"] is True
    assert spec["includeAll"] is True
    assert spec["allValue"] == ".*"


def test_host_panels_should_be_labelled_node_level(
    trainer_dashboard: dict[str, Any],
) -> None:
    # The host metrics describe the whole node, so the panel must not imply
    # they belong to a single job on a shared node.
    for panel_id in _HOST_PANELS:
        panel = trainer_dashboard["elements"][f"panel-{panel_id}"]["spec"]
        assert panel["title"].startswith("Node "), panel["title"]
        description = panel["description"]
        assert "not only this training job" in description or (
            "other processes" in description
        ), description


@pytest.mark.parametrize(
    ("panels", "legend", "shows_card"),
    [
        (_RAY_PROCESS_PANELS, _RAY_PROCESS_LEGEND, True),
        (_SELF_PROCESS_PANELS, _SELF_PROCESS_LEGEND, False),
    ],
    ids=["ray-actor-process", "self-process"],
)
def test_process_panels_should_show_in_process_metrics(
    trainer_dashboard: dict[str, Any],
    panels: dict[str, str],
    legend: str,
    shows_card: bool,
) -> None:
    # The process collectors report in-process metrics only, so the panels
    # carry a single query. They are process-level (not node-level): the legend
    # separates the reporting processes of a shared node, and the Ray worker
    # ones add the card so a curve can be traced past its pid.
    for panel_id, family in panels.items():
        queries = _panel_queries(trainer_dashboard, panel_id)
        assert queries, panel_id
        for expr, legend_format, hidden in queries:
            assert not hidden, expr
            assert f"{_METRIC_PREFIX}{family}" in expr, expr
            assert 'project=~"$project"' in expr, expr
            assert 'experiment_name=~"$experiment_name"' in expr, expr
            assert 'worker_role=~"$worker_role"' in expr, expr
            assert _device_filter_selects(expr) is shows_card, expr
            assert legend_format == legend, legend_format


def test_device_variable_should_filter_workers_and_cards_together(
    trainer_dashboard: dict[str, Any],
) -> None:
    variable = next(
        variable
        for variable in trainer_dashboard["variables"]
        if variable["spec"]["name"] == "device_index"
    )
    spec = variable["spec"]

    assert f"{_METRIC_PREFIX}" in spec["definition"]
    assert "device_index" in spec["definition"]
    assert spec["multi"] is True
    assert spec["includeAll"] is True
    assert spec["allValue"] == ".*"
    # A query variable without an executable query resolves to no value at all,
    # which blanks every panel carrying the filter.
    assert spec["query"]["spec"]["query"] == spec["definition"]
    assert spec["query"]["datasource"] == {"name": "${datasource}"}
    assert spec["refresh"] == "onDashboardLoad"
    assert spec["current"] == {"text": "All", "value": "$__all"}

    # One dropdown drives both the worker processes and the cards they run on,
    # so a worker can be read next to its own NPU/GPU panels.
    filtered = [
        panel_id
        for panel_id in (*_NPU_DEVICE_PANELS, *_GPU_DEVICE_PANELS, *_RAY_PROCESS_PANELS)
        if all(
            _device_filter_selects(expr)
            for expr in _visible(trainer_dashboard, panel_id)
        )
    ]
    assert filtered == [
        *_NPU_DEVICE_PANELS,
        *_GPU_DEVICE_PANELS,
        *_RAY_PROCESS_PANELS,
    ]


def test_only_ray_process_legends_should_sort_by_max_descending(
    trainer_dashboard: dict[str, Any],
) -> None:
    # A single four-card run already spawns eight storage/reward/agent workers,
    # so those legends rank by utilization to surface the busiest series. Every
    # other hardware panel keeps its natural series order: ranking NPU/GPU
    # legends by utilization would shuffle the cards out of rank order.
    for panel_id in _HW_TIMESERIES_PANELS:
        element = trainer_dashboard["elements"].get(f"panel-{panel_id}")
        if element is None:
            continue
        viz = element["spec"]["vizConfig"]
        if viz.get("group") != "timeseries":
            continue
        legend = viz["spec"]["options"]["legend"]
        if panel_id in _RAY_PROCESS_PANELS:
            assert legend["sortBy"] == "Max", panel_id
            assert legend["sortDesc"] is True, panel_id
        else:
            assert "sortBy" not in legend, panel_id
            assert "sortDesc" not in legend, panel_id


def test_hardware_panels_should_filter_by_selected_node(
    trainer_dashboard: dict[str, Any],
) -> None:
    # Every node-scoped panel separates one node's series from the rest, so a
    # node tab shows a single node.
    for panel_id in _NODE_SCOPED_PANELS:
        for expr, _, _ in _panel_queries(trainer_dashboard, panel_id):
            assert _METRIC_PREFIX in expr, expr
            assert _node_filter_selects(expr), expr


def test_hardware_rows_should_not_repeat_as_node_tabs(
    trainer_dashboard: dict[str, Any],
) -> None:
    # Repeating a row per node needs the node variable to resolve while the
    # dashboard loads. ``query_result`` is evaluated at the end of the time
    # range, so a finished run (no samples in the last lookback window) yields no
    # nodes, and the row collapses into a single empty ``All`` tab that renders
    # nothing. The rows therefore stay flat and node switching goes through the
    # ``hardware: Node`` picker, which also keeps a 1024-card cluster from
    # spawning hundreds of tabs.
    expected_items = {
        "CPU / Host Metrics": (325, 326, 328, 344, 345, 346, 347, 348),
        "Ascend NPU Metrics": (329, 330, 331, 332, 333, 334, 335),
    }
    for title, panel_ids in expected_items.items():
        row = _find_row_spec(trainer_dashboard["layout"], title)
        assert row is not None, title
        layout = row["layout"]
        assert layout["kind"] == "AutoGridLayout", title
        # ``json.dumps`` catches a nested repeat on a tab too.
        assert "repeat" not in json.dumps(layout), title
        names = [item["spec"]["element"]["name"] for item in layout["spec"]["items"]]
        expected = [
            f"panel-{panel_id}"
            for panel_id in panel_ids
            if f"panel-{panel_id}" in trainer_dashboard["elements"]
        ]
        assert names == expected, title


def test_dashboards_should_not_reference_the_removed_instance_variable(
    trainer_dashboard: dict[str, Any],
) -> None:
    # The node variable replaced the old instance-scoped one everywhere; a
    # leftover reference in a panel query would silently blank that panel.
    assert "npu_instance" not in json.dumps(trainer_dashboard)


def test_hardware_panels_should_not_query_legacy_exporters(
    trainer_dashboard: dict[str, Any],
) -> None:
    # The built-in collector is the only hardware source; a leftover exporter
    # query would draw a second, unrelated series in the same panel.
    text = json.dumps(trainer_dashboard)
    for marker in ("node-exporter", "npu-exporter", "DCGM_FI_DEV_"):
        assert marker not in text, marker


# Descriptions are tooltips, not documentation: keep them one line long.
_MAX_DESCRIPTION = 120
_REMOVED_DESCRIPTION_TAILS = ("node_exporter", "DCGM exporter", "in-process monitor or")


def _find_row_spec(layout: Any, title: str) -> dict[str, Any] | None:
    """The spec of the row titled ``title`` anywhere in the layout tree."""
    if not isinstance(layout, dict):
        return None
    spec = layout.get("spec")
    if not isinstance(spec, dict):
        return None
    kind = layout.get("kind")
    if kind == "RowsLayout":
        for row in spec.get("rows") or []:
            row_spec = row.get("spec") or {}
            if row_spec.get("title") == title:
                return row_spec
            found = _find_row_spec(row_spec.get("layout"), title)
            if found is not None:
                return found
    elif kind == "TabsLayout":
        for tab in spec.get("tabs") or []:
            found = _find_row_spec((tab.get("spec") or {}).get("layout"), title)
            if found is not None:
                return found
    return None


def test_device_rows_should_always_render(
    trainer_dashboard: dict[str, Any],
) -> None:
    # A data-derived rule blanks these rows exactly when the vendor's metrics go
    # stale, so both rows render unconditionally and their panels report the
    # missing data themselves. The probes that used to feed such a rule are gone.
    variables = {
        variable["spec"]["name"] for variable in trainer_dashboard["variables"]
    }
    for title in ("Ascend NPU Metrics", "NVIDIA GPU Metrics"):
        row = _find_row_spec(trainer_dashboard["layout"], title)
        assert row is not None, title
        assert "conditionalRendering" not in row, title
    assert "npu_detected" not in variables
    assert "gpu_detected" not in variables


def test_hardware_descriptions_should_stay_short(
    trainer_dashboard: dict[str, Any],
) -> None:
    panel_ids = (
        *_INTEGRATED_PANELS,
        "344",
        *_RAY_PROCESS_PANELS,
        *_SELF_PROCESS_PANELS,
    )
    for panel_id in panel_ids:
        element = trainer_dashboard["elements"].get(f"panel-{panel_id}")
        if element is None:
            continue  # the omni dashboard has no disk panel
        description = element["spec"].get("description", "")
        assert description, panel_id
        assert len(description) <= _MAX_DESCRIPTION, (panel_id, description)
        for tail in _REMOVED_DESCRIPTION_TAILS:
            assert tail not in description, (panel_id, description)
