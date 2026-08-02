#!/usr/bin/env bash
#
# Download fitted models, feature matrices, keypoint-MoSeq outputs, and cleaned
# CalMS21 keypoints from the companion Zenodo record.
#
# Usage:
#   bash features/fetch_derived_bundle.sh derived
#   bash features/fetch_derived_bundle.sh moseq
#   bash features/fetch_derived_bundle.sh all --dry-run

set -euo pipefail

# Companion Zenodo record ID. The legacy name remains accepted so an existing
# local command does not break during the project-name transition.
ZENODO_RECORD="${SOCIAL_DLDS_ZENODO_RECORD:-${DLDS_ZENODO_RECORD:-}}"

BUNDLE="${1:-}"
shift || true

DRY_RUN=0
DEST_OVERRIDE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)          DRY_RUN=1 ;;
        --dest)             DEST_OVERRIDE="$2"; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

case "$BUNDLE" in
    derived|features|moseq|keypoints|all) ;;
    *)
        cat >&2 <<'EOF'
usage: fetch_derived_bundle.sh {derived|features|moseq|keypoints|all} [options]

  --dry-run           list what would be downloaded, download nothing
  --dest DIR          override the destination data root
EOF
        exit 2
        ;;
esac

# Resolve the destination with the same path configuration as the Python code.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }

resolve_dest() {
    if [ -n "$DEST_OVERRIDE" ]; then echo "$DEST_OVERRIDE"; return; fi
    PYTHONPATH="$REPO" python3 -c 'from dlds_release.paths import ROOT; print(ROOT)'
}

DEST="$(resolve_dest)"

if [ -z "$ZENODO_RECORD" ]; then
    cat >&2 <<EOF
No companion Zenodo record is configured.
Run:

  SOCIAL_DLDS_ZENODO_RECORD=<id> bash features/fetch_derived_bundle.sh $BUNDLE
EOF
    exit 1
fi

API="https://zenodo.org/api/records/${ZENODO_RECORD}"

case "$BUNDLE" in
    derived)   WANTED="dlds_derived_models.tar.zst" ;;
    features)  WANTED="dlds_feature_inputs.tar.zst" ;;
    moseq)     WANTED="kpmoseq_outputs.tar.zst" ;;
    keypoints) WANTED="calms21_cleaned_keypoints.tar.zst" ;;
    all)       WANTED="dlds_derived_models.tar.zst dlds_feature_inputs.tar.zst kpmoseq_outputs.tar.zst calms21_cleaned_keypoints.tar.zst" ;;
esac
echo "record      : $ZENODO_RECORD"
echo "destination : $DEST"
echo "files       : $WANTED"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "(dry run, nothing downloaded)"
    exit 0
fi

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }

mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

curl -sSL "$API" -o "$TMP/record.json"

for name in $WANTED; do
    # Ask the API for this file's URL and the publisher's checksum, rather than
    # constructing a URL: Zenodo mints a new one per record version.
    read -r url checksum < <(python3 - "$TMP/record.json" "$name" <<'PY'
import json, sys
record, want = sys.argv[1], sys.argv[2]
for f in json.load(open(record)).get("files", []):
    if f.get("key") == want:
        print(f["links"]["self"], f.get("checksum", ""))
        break
else:
    sys.exit(f"file not present in record: {want}")
PY
    )

    out="$TMP/$name"
    echo "==> $name"
    curl -# -L "$url" -o "$out"

    if [ -n "$checksum" ]; then
        algo="${checksum%%:*}"
        want_hex="${checksum#*:}"
        got_hex="$(python3 -c "
import hashlib,sys
h=hashlib.new(sys.argv[2])
with open(sys.argv[1],'rb') as fh:
    for b in iter(lambda: fh.read(1<<22), b''): h.update(b)
print(h.hexdigest())" "$out" "$algo")"
        if [ "$got_hex" != "$want_hex" ]; then
            echo "checksum mismatch for $name: expected $want_hex, got $got_hex" >&2
            exit 1
        fi
        echo "    $algo ok"
    fi

    case "$name" in
        *.tar.zst)
            command -v zstd >/dev/null || { echo "zstd is required to unpack $name" >&2; exit 1; }
            tar --use-compress-program=unzstd -xf "$out" -C "$DEST"
            ;;
        *)
            mv "$out" "$DEST/"
            ;;
    esac
done

echo
echo "unpacked into $DEST"
echo "check the path contract resolves:  python scripts/check_env.py"
