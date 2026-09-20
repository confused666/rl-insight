"""Check readiness of every bundled service without requiring curl."""

from urllib.request import ProxyHandler, build_opener

from rl_insight.server.network import format_host_port, local_addresses

opener = build_opener(ProxyHandler({}))
host = local_addresses()["loopback"]
for port, path in (
    (18080, "/healthz"),
    (9090, "/-/ready"),
    (3200, "/ready"),
    (3000, "/api/health"),
):
    with opener.open(
        f"http://{format_host_port(host, port)}{path}", timeout=3
    ) as response:
        if response.status != 200:
            raise SystemExit(f"Service on port {port} is not ready")
