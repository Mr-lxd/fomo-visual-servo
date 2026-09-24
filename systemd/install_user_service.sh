#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="robobeetle-vision.service"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_UNIT="${SCRIPT_DIR}/${SERVICE_NAME}"
DEST_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
DEST_UNIT="${DEST_DIR}/${SERVICE_NAME}"

RUNTIME_DIR="${HOME}/fomo-vision-runtime"
PYTHON_BIN="${HOME}/venvs/fomo-ort-d2-epoch40/bin/python"
MODEL_PATH="${RUNTIME_DIR}/artifacts/d2_mobilenet_v2_fomo_seed42_epoch40.onnx"
SIDECAR_PATH="${RUNTIME_DIR}/artifacts/d2_mobilenet_v2_fomo_seed42_epoch40.onnx.json"
CAPTURE_ROOT="${RUNTIME_DIR}/datasets_raw"

start_now=false

usage() {
    cat <<'EOF'
Usage: ./systemd/install_user_service.sh [--now]

Installs and enables the RoboBeetle Vision user service.

Options:
  --now   Start or restart the service after installation.
  -h, --help
          Show this help.

Verified runtime layout:
  ~/fomo-vision-runtime
  ~/fomo-vision-data
  ~/venvs/fomo-ort-d2-epoch40
EOF
}

while (($#)); do
    case "$1" in
        --now)
            start_now=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

require_path() {
    local kind="$1"
    local path="$2"
    if [[ ! -e "$path" ]]; then
        echo "Missing ${kind}: ${path}" >&2
        exit 1
    fi
}

require_path "runtime launcher" "${RUNTIME_DIR}/run.py"
require_path "runtime Python" "${PYTHON_BIN}"
require_path "ONNX model" "${MODEL_PATH}"
require_path "ONNX sidecar" "${SIDECAR_PATH}"
require_path "capture/data root" "${CAPTURE_ROOT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Runtime Python is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"
install -m 0644 "${SOURCE_UNIT}" "${DEST_UNIT}"

systemctl --user daemon-reload
systemd-analyze --user verify "${DEST_UNIT}"
systemctl --user enable "${SERVICE_NAME}"

linger="$(loginctl show-user "${USER}" -p Linger --value 2>/dev/null || true)"
if [[ "${linger}" != "yes" ]]; then
    cat >&2 <<EOF
WARNING: systemd user lingering is not enabled for ${USER}.
The service is enabled, but unattended boot startup requires:
  sudo loginctl enable-linger ${USER}
EOF
fi

if [[ "${start_now}" == "true" ]]; then
    if systemctl --user is-active --quiet "${SERVICE_NAME}"; then
        systemctl --user restart "${SERVICE_NAME}"
    else
        if pgrep -af '[r]un.py vision_live' >/tmp/robobeetle-vision-unmanaged-processes.txt; then
            echo "Refusing to start: an unmanaged Vision LIVE process is already running:" >&2
            cat /tmp/robobeetle-vision-unmanaged-processes.txt >&2
            rm -f /tmp/robobeetle-vision-unmanaged-processes.txt
            exit 3
        fi
        rm -f /tmp/robobeetle-vision-unmanaged-processes.txt
        systemctl --user start "${SERVICE_NAME}"
    fi

    if ! systemctl --user is-active --quiet "${SERVICE_NAME}"; then
        echo "Vision service failed to become active." >&2
        systemctl --user status "${SERVICE_NAME}" --no-pager >&2 || true
        exit 4
    fi
fi

echo "Installed: ${DEST_UNIT}"
echo "Enabled:   $(systemctl --user is-enabled "${SERVICE_NAME}")"
if [[ "${start_now}" == "true" ]]; then
    echo "Active:    $(systemctl --user is-active "${SERVICE_NAME}")"
fi

echo
echo "Useful commands:"
echo "  systemctl --user status ${SERVICE_NAME}"
echo "  systemctl --user restart ${SERVICE_NAME}"
echo "  systemctl --user stop ${SERVICE_NAME}"
echo "  systemctl --user disable ${SERVICE_NAME}"
