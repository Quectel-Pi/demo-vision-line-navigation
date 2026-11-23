#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
app_user="${SUDO_USER:-$(id -un)}"
python_bin="${project_dir}/.venv/bin/python"
service_file="/etc/systemd/system/line-follow-demo.service"

if [[ "${project_dir}" == *" "* ]]; then
    echo "Project path must not contain spaces: ${project_dir}" >&2
    exit 1
fi

if [[ ! -x "${python_bin}" ]]; then
    echo "Virtual environment not found: ${python_bin}" >&2
    echo "Create .venv and install requirements.txt before enabling autostart." >&2
    exit 1
fi

if ! id "${app_user}" >/dev/null 2>&1; then
    echo "User does not exist: ${app_user}" >&2
    exit 1
fi

if ! getent group dialout >/dev/null || ! getent group video >/dev/null; then
    echo "Required dialout/video groups are unavailable on this system." >&2
    exit 1
fi

sudo -v
sudo tee "${service_file}" >/dev/null <<UNIT
[Unit]
Description=Vision line-following demo
After=local-fs.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=${app_user}
SupplementaryGroups=dialout video
WorkingDirectory=${project_dir}
Environment=PYTHONUNBUFFERED=1
ExecStart=${python_bin} -m line_follow_demo.web_app --config config/demo.json --host 0.0.0.0 --port 8081 --auto-start
Restart=on-failure
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now line-follow-demo.service
sudo systemctl --no-pager --full status line-follow-demo.service
