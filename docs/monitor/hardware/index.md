# Hardware Monitoring

RL-Insight collects CPU, memory, network, disk, and accelerator (NVIDIA GPU /
Ascend NPU) metrics from inside the training job. The monitor starts with
`rl_insight.init()`, so there is no exporter to install, no port to expose, and
no Prometheus target to register.

| Guide | Contents |
|---|---|
| [中文](hardware_monitoring_zh.md) | Configuration, execution flow, metrics, and troubleshooting |
| [English](hardware_monitoring_en.md) | Configuration, execution flow, metrics, and troubleshooting |

The hardware row of the trainer dashboards (for example
`verl_trainer_v1_with_vllm_engine`) reads these in-process metrics directly.
Hardware is enabled by default; set `hardware.enabled` to `False` or
`RL_INSIGHT_HARDWARE_ENABLED=0` to turn collection off:

```python
rl_insight.init(config={"hardware": {"enabled": False}})
```

`hardware.interval_seconds` controls the sampling period (default `10`,
minimum `5`). In a Ray job, one monitor runs per node, so host metrics are
collected once per node regardless of how many workers call `init()`.
