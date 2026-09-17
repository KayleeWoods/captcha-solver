#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/tools/captcha-solver
export PORT=8877
export TURNSTILE_HEADLESS=0
export TURNSTILE_MAX_CONCURRENT=2
export CLOAKBROWSER_SUPPRESS_FONT_WARNING=1
# optional sticky proxy for solves
# export TURNSTILE_PROXY="http://user:pass@host:port"
source .venv/bin/activate
# 1280x720x16 drops virtual framebuffer and CPU software rasterizer load by ~60% compared to 1080p 24-bit
exec xvfb-run -a --server-args="-screen 0 1280x720x16" python3 server.py
