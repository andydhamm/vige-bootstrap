# vige-bootstrap

Bootstrap scripts + multi-part app tarball chunks for **ViGe burst-workers** on vast.ai.

## Purpose

Lets the ViGe burst manager rent vast.ai GPU instances using a stock
`vastai/pytorch` base image (which boots reliably on vast hosts) and
layer the ViGe inference stack on top at first boot — instead of
publishing a custom 14 GB docker image that fights vast's instance
daemon and stalls on flaky hosts.

## Layout

- `vast_bootstrap.sh` — runs on the vast box via `--onstart-cmd`, downloads
  the chunks from a GitHub Release, reassembles + extracts `/app`,
  `/opt/worker`, `/workspace/LightX2V`. Idempotent.
- Releases (`app-YYYYMMDD-HHMM`) — multi-part `*.tar.zst.part-*` chunks
  (each <2 GB) + `sha256sums.txt` + `manifest.json`. Produced by
  `vige/scripts/publish_app_tarball.sh` on cgyws1.

## How vast.ai uses it

The burst manager (`vige_burst/providers/vastai.py`) issues
`vastai create instance` with:

```
--image vastai/pytorch:cuda-13.0.2-auto
--onstart-cmd 'curl -sfL https://raw.githubusercontent.com/andydhamm/vige-bootstrap/main/vast_bootstrap.sh | bash'
--env '-e VIGE_APP_TAG=app-YYYYMMDD-HHMM ...'
```

The bootstrap runs **after** vast's default ssh-launch setup (apt + sshd
+ account-key auth), then downloads + extracts the latest tagged app
tarball, ~3-5 min on a fast-pipe host.

## Republishing

Re-build the slim image locally, push to ghcr, then on cgyws1:

```
IMAGE=ghcr.io/andydhamm/vige-worker-infer-slim:latest \
  RELEASE_TAG=app-$(date -u +%Y%m%d-%H%M) \
  bash vige/scripts/publish_app_tarball.sh
```

Update `VIGE_APP_TAG` in the burst profile to the new tag.
