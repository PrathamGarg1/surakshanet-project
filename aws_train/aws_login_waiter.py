#!/usr/bin/env python3
"""Remote aws login helper for headless cloud VMs.

1) Starts `aws login --remote`
2) Writes the authorize URL to /tmp/aws-login/url
3) Waits for the user-pasted code in /tmp/aws-login/code
4) Submits it and exits
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

DIR = Path("/tmp/aws-login")
DIR.mkdir(parents=True, exist_ok=True)
STATUS = DIR / "status"
URL_PATH = DIR / "url"
CODE_PATH = DIR / "code"
LOG = DIR / "waiter.log"


def log(msg: str) -> None:
    line = msg.rstrip() + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def main() -> int:
    if CODE_PATH.exists():
        CODE_PATH.unlink()
    STATUS.write_text("starting\n")

    env = os.environ.copy()
    env["AWS_DEFAULT_REGION"] = env.get("AWS_DEFAULT_REGION", "us-east-1")
    env["AWS_REGION"] = env.get("AWS_REGION", "us-east-1")

    proc = subprocess.Popen(
        ["aws", "login", "--remote", "--region", env["AWS_REGION"]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None and proc.stdin is not None

    url = None
    deadline = time.time() + 90
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            log(line.rstrip("\n"))
            m = re.search(r"https://[^\s]+", line)
            if m and "authorize" in m.group(0):
                url = m.group(0).rstrip(").,]")
                URL_PATH.write_text(url + "\n")
                STATUS.write_text("waiting_for_code\n")
                log(f"URL_SAVED={url}")
                break

    if not url:
        STATUS.write_text("failed_no_url\n")
        rest = proc.communicate(timeout=10)[0] or ""
        log(rest)
        return 1

    log("Waiting up to 10 minutes for /tmp/aws-login/code ...")
    wait_deadline = time.time() + 600
    while time.time() < wait_deadline:
        if CODE_PATH.exists():
            code = CODE_PATH.read_text(encoding="utf-8").strip()
            if code:
                proc.stdin.write(code + "\n")
                proc.stdin.flush()
                STATUS.write_text("code_submitted\n")
                log("code submitted")
                break
        time.sleep(1)
    else:
        STATUS.write_text("timeout_waiting_code\n")
        proc.kill()
        return 2

    try:
        rest = proc.communicate(timeout=180)[0] or ""
    except subprocess.TimeoutExpired:
        proc.kill()
        STATUS.write_text("timeout_after_code\n")
        return 3
    log(rest)
    rc = proc.returncode or 0
    STATUS.write_text(f"done:{rc}\n")
    CODE_PATH.unlink(missing_ok=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
