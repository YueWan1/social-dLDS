#!/usr/bin/env bash
#
# Stage 0b: CalMS21, Task 1 "classic classification"  (CaltechDATA DOI 10.22002/D1.1991)
#
# This is the dyadic half of the paper: 70 resident-intruder sessions, MARS
# 7-keypoint tracking of both mice at 30 fps, with frame-level annotations by a
# single expert (annotator 1) into attack / investigation / mount / other.
#
# By default this takes three files:
#
#   task1_classic_classification.zip   457 MB  the train+test JSON, what we need
#   readme.md                           11 KB  the authoritative schema description
#   calms21_convert_to_npy.py          4.4 KB  the publisher's own JSON->npy helper
#
# and, only with --videos, the 26 GB `task1_videos_mp4.zip`.  The videos are not
# part of the analysis: they are read solely by the frame-overlay and filmstrip
# panels, which need <raw_root>/calms21/task1_videos_mp4/train/mouse<NNN>_task1_annotator1.mp4.
#
# The JSON is NOT the layout the feature builder reads.  Run
#   python features/calms21_to_npy.py
# afterwards; that step produces <raw_root>/calms21_npy/.
#
# Files are enumerated through the CaltechDATA (InvenioRDM) REST API rather than
# hard-coded, both so a re-versioned deposit still resolves and because the API
# returns the publisher's MD5.  Note that the human-facing record page at
# https://data.caltech.edu/records/s0vdx-0k302 answers 403 to non-browser
# clients, while /api/records/... does not; that asymmetry is why this script
# never touches the HTML page.
#
# Usage:
#   bash features/fetch_calms21.sh                # keypoints + annotations
#   bash features/fetch_calms21.sh --videos       # also the 26 GB mp4 archive
#   bash features/fetch_calms21.sh --dry-run      # show what would be fetched
#   bash features/fetch_calms21.sh --dest DIR     # override the destination
#   bash features/fetch_calms21.sh --only readme.md
#                                                # one file by name; the cheap way
#                                                # to check the transfer and the
#                                                # MD5 check before committing to
#                                                # a 457 MB download
#
# Licence: CC-BY-4.0 (as recorded by CaltechDATA).  Cite Sun et al., "The
# Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions", NeurIPS 2021
# Datasets and Benchmarks.  Nothing is redistributed here.

set -euo pipefail

RECORD=s0vdx-0k302
API="https://data.caltech.edu/api/records/${RECORD}"

CORE_FILES="task1_classic_classification.zip readme.md calms21_convert_to_npy.py"
VIDEO_FILES="task1_videos_mp4.zip"

# Resolve the destination with the same path configuration as the Python code.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

raw_root() {
  PYTHONPATH="$REPO" python3 -c 'from dlds_release.paths import RAW_ROOT; print(RAW_ROOT)'
}

DRY_RUN=0
WITH_VIDEOS=0
DEST=""
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --videos) WITH_VIDEOS=1 ;;
    --only) ONLY="${2:?--only needs a file name}"; shift ;;
    --dest) DEST="${2:?--dest needs a directory}"; shift ;;
    -h|--help) sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

command -v python3 >/dev/null || { echo "missing required tool: python3" >&2; exit 1; }
[ -n "$DEST" ] || DEST="$(raw_root)/calms21"

for tool in curl unzip; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done

md5_of() {
  if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1
  else md5 -q "$1"; fi
}

WANTED="$CORE_FILES"
[ "$WITH_VIDEOS" = 1 ] && WANTED="$WANTED $VIDEO_FILES"
[ -n "$ONLY" ] && WANTED="$ONLY"

echo "record      : ${API}"
echo "destination : ${DEST}"

manifest="$(curl -fsSL --retry 3 -H 'Accept: application/json' "$API" | python3 -c '
import json, sys
rec = json.load(sys.stdin)
rights = rec["metadata"].get("rights") or [{}]
print("#license", rights[0].get("id", "unknown"), sep="\t")
for key, entry in sorted(rec["files"]["entries"].items()):
    print(key, entry.get("size", 0), entry.get("checksum", ""), sep="\t")
')"

echo "licence     : $(printf '%s\n' "$manifest" | awk -F'\t' '$1=="#license"{print $2}')"
echo
printf '%s\n' "$manifest" | awk -F'\t' -v want=" $WANTED " '$1!~/^#/ {
  printf "  %-36s %13d  %s\n", $1, $2, (index(want, " " $1 " ") ? "TAKE" : "skip")
}'
echo

[ "$DRY_RUN" = 1 ] || mkdir -p "$DEST"

for key in $WANTED; do
  line="$(printf '%s\n' "$manifest" | awk -F'\t' -v k="$key" '$1==k')"
  [ -n "$line" ] || { echo "'$key' is not in record $RECORD -- has the deposit changed?" >&2; exit 1; }
  size=$(printf '%s\n' "$line" | cut -f2)
  want_md5=$(printf '%s\n' "$line" | cut -f3); want_md5="${want_md5#md5:}"
  url="${API}/files/${key}/content"
  out="$DEST/$key"

  if [ "$DRY_RUN" = 1 ]; then
    echo "DRY RUN. would download $url"
    echo "   -> $out   ($size bytes, md5 $want_md5)"
    continue
  fi

  echo ">> $key ($size bytes)"
  curl -fL --retry 3 --retry-delay 5 -C - -o "$out" "$url"
  got_md5="$(md5_of "$out")"
  if [ "$got_md5" != "$want_md5" ]; then
    echo "MD5 mismatch for $out" >&2
    echo "  expected $want_md5" >&2
    echo "  got      $got_md5" >&2
    exit 1
  fi
  echo "   md5 ok: $got_md5"
  case "$key" in
    *.zip) unzip -q -o "$out" -x '__MACOSX/*' -d "$DEST" ;;
  esac
done

if [ "$DRY_RUN" = 1 ]; then
  echo
  echo "then: unzip the archives into $DEST/ and run features/calms21_to_npy.py"
  exit 0
fi

train_json="$(find "$DEST" -name 'calms21_task1_train.json' -print -quit)"
echo
echo "extracted to $DEST"
echo "  task1 train json : ${train_json:-MISSING}"
if [ "$WITH_VIDEOS" = 1 ]; then
  n_mp4=$(find "$DEST" -name 'mouse*_task1_annotator1.mp4' | wc -l)
  echo "  videos           : $n_mp4 (expected 70)"
fi

cat <<'NEXT'

Next:
  python features/calms21_to_npy.py            # JSON -> <raw_root>/calms21_npy/
  python features/build_feature27_dyadic.py --kp-sigma 1.0 --pose-sigma 0.5
NEXT
