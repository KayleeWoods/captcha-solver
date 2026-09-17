#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/tools/captcha-solver
export PORT=8877
export TURNSTILE_HEADLESS=0
export CLOAKBROWSER_SUPPRESS_FONT_WARNING=1
# optional sticky proxy for solves
# export TURNSTILE_PROXY="http://user:pass@host:port"
source .venv/bin/activate
exec xvfb-run -a --server-args="-screen 0 1920x1080x24" python3 server.py
