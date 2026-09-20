"""Run the foreground CLI with Docker stop and restart semantics."""

import signal
import sys

from rl_insight.cli import main
from rl_insight.server.catalog import DEFAULT_STATE_ROOT, STATE_FILE

# Docker sends SIGTERM; the foreground CLI already cleans up on Ctrl+C.
signal.signal(signal.SIGTERM, signal.default_int_handler)
signal.signal(signal.SIGINT, signal.default_int_handler)

# A stopped container can retain stale PIDs which are reused after restart.
# This entrypoint owns a fresh container PID namespace; data is kept separately.
(DEFAULT_STATE_ROOT / "run" / STATE_FILE).unlink(missing_ok=True)
sys.exit(main())
