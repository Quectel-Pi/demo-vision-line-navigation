#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${project_dir}/.venv/bin/python"
service_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
service_name="line-follow-preview.service"
service_file="${service_dir}/${service_name}"

if [[ "${project_dir}" == *" "* ]]; then
    echo "Project path must not contain spaces: ${project_dir}" >&2
    exit 1
fi

if [[ ! -x "${python_bin}" ]]; then
    echo "Virtual environment not found: ${python_bin}" >&2
    echo "Create .venv and install requirements.txt before enabling autostart." >&2
    exit 1
fi

user_groups=" $(id -nG) "
for group in dialout video; do
    if [[ "${user_groups}" != *" ${group} "* ]]; then
        echo "Current user must belong to the ${group} group." >&2
        exit 1
    fi
done

mkdir -p "${service_dir}"
cat > "${service_file}" <<UNIT
[Unit]
Description=Vision line-following preview and camera gimbal homing
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=${project_dir}
Environment=PYTHONUNBUFFERED=1
ExecStart=${python_bin} -m line_follow_demo --config config/demo.json --host 0.0.0.0 --port 8081
Restart=on-failure
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=10

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable --now "${service_name}"
systemctl --user --no-pager --full status "${service_name}"

echo "Preview-only service installed. It does not arm wheel tracking."
