#!/usr/bin/env bash
# ViGe burst-worker bootstrap, run via vast.ai --onstart-cmd.
#
# Vast.ai provides a working base (sshd + portal + key auth) via their
# pytorch/base templates. This script runs after their default onstart
# and layers our pre-built /app + worker code on top, by downloading a
# multi-part tarball from a GitHub Release.
#
# Idempotent: skips if /app/.vige-ready already exists (so stop→start
# of the same instance is instant).
#
# Knobs (env vars set in vast's --env or onstart-cmd):
#   VIGE_BOOTSTRAP_REPO  gh repo holding the release (default: andydhamm/vige-bootstrap)
#   VIGE_APP_TAG         release tag to pull (default: app-latest)
#   VIGE_BOOTSTRAP_PARALLEL  parallel chunk downloads (default: 6)
set -euo pipefail

log() { printf '[vige-bootstrap] %s\n' "$*" >&2; }
die() { log "FATAL: $*"; exit 1; }

REPO="${VIGE_BOOTSTRAP_REPO:-andydhamm/vige-bootstrap}"
TAG="${VIGE_APP_TAG:-app-latest}"
PARALLEL="${VIGE_BOOTSTRAP_PARALLEL:-6}"
SENTINEL=/app/.vige-ready

# --- 0. Idempotent skip --------------------------------------------
if [[ -f "$SENTINEL" ]]; then
    log "$SENTINEL exists — skipping bootstrap"
    exit 0
fi

log "starting bootstrap from $REPO@$TAG"
BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"

# --- 1. Runtime libs (parallel with downloads) ---------------------
{
    apt-get update >/dev/null 2>&1 || log "WARN: apt-get update failed"
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 zstd \
        >/dev/null 2>&1 || log "WARN: apt-get install failed; most libs may already be present"
} &
APT_PID=$!

# --- 2. Stage + download manifest + chunks --------------------------
STAGE=/opt/vige-bootstrap
mkdir -p "$STAGE" && cd "$STAGE"

log "fetching sha256sums.txt"
curl -fsSL --retry 3 "${BASE_URL}/sha256sums.txt" -o sha256sums.txt \
    || die "could not fetch sha256sums.txt from ${BASE_URL}"

N_CHUNKS=$(wc -l < sha256sums.txt)
TOTAL_BYTES_HUMAN=$(awk '{print $2}' sha256sums.txt | head)
log "manifest: ${N_CHUNKS} chunks"

# Download all chunks in parallel. The release URLs are stable so retries
# are safe; curl --retry handles transient 5xx.
log "downloading ${N_CHUNKS} chunks (parallel=${PARALLEL})"
awk '{print $2}' sha256sums.txt \
    | xargs -I{} -P "${PARALLEL}" \
        curl -fsSL --retry 3 --retry-delay 2 -o "{}" "${BASE_URL}/{}"

# --- 3. Verify checksums --------------------------------------------
log "verifying sha256 checksums"
sha256sum -c sha256sums.txt > /dev/null \
    || die "checksum mismatch — chunks corrupted, aborting"

# --- 4. Wait on apt (probably done) + ensure zstd is present --------
wait "$APT_PID" 2>/dev/null || true
command -v zstd >/dev/null || die "zstd not installed and apt failed"

# --- 5. Reassemble + extract to / ----------------------------------
log "reassembling + extracting to /"
# tar -x runs as PID-1 child; extraction will create /app, /opt/worker,
# /workspace/LightX2V. Existing files in those paths get overwritten —
# that's expected for fresh boots.
cat $(awk '{print $2}' sha256sums.txt | sort) \
    | zstd -d \
    | tar -xC /

# --- 6. Cleanup chunks (free ~12 GB) -------------------------------
log "cleaning up chunks"
rm -f *.part-* sha256sums.txt

# --- 7. /etc/profile.d for interactive ssh sessions ----------------
cat > /etc/profile.d/vige.sh <<'EOF'
export PYTHONPATH=/workspace/LightX2V:/opt/worker/code${PYTHONPATH:+:$PYTHONPATH}
export PATH=/app/miniconda/bin:$PATH
EOF
chmod 644 /etc/profile.d/vige.sh

# --- 8. Sanity check + sentinel ------------------------------------
/app/miniconda/bin/python -c "
import torch
print(f'[vige-bootstrap] torch {torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
" 2>&1 | sed 's/^/[vige-bootstrap] /' >&2 \
    || log "WARN: torch sanity check failed (continuing anyway — gpu may not be visible at bootstrap time)"

touch "$SENTINEL"
log "DONE — sentinel: $SENTINEL"
