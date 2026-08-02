#!/usr/bin/env bash
#
# Stage 0a: keypoint-MoSeq open-field 2D dataset  (Zenodo record 15171335)
#
# This is the single-mouse half of the paper: 10 top-down IR sessions of a
# freely-moving mouse, tracked with DeepLabCut.  Only `open_field_2D.zip` is
# taken; the record's other three files (dopamine, accelerometry, open_field_3D,
# ~1.9 GB together) are never touched by any script in this repository.
#
# After extraction the tree is
#
#   <raw_root>/keypoint_moseq_zenodo_15171335/open_field_2D/
#       config.yaml
#       videos/*.h5     10 DeepLabCut tracks   <- features/build_feature16_single_mouse.py
#       videos/*.mp4    10 companion videos    <- only the Fig-1a/1e clip renders
#
# The .mp4 ship inside the same zip, so there is no way to skip them at
# download time; they are about 110 MB of the 288 MB and can be removed
# afterwards if you do not intend to re-render the Figure 1 filmstrip.
#
# Files are enumerated through the Zenodo REST API rather than by hard-coding
# a download URL: Zenodo mints a new URL per record version, and the API also
# hands back the publisher's MD5, which is what this script verifies against.
#
# Usage:
#   bash features/fetch_zenodo_kpms.sh              # download, verify, extract
#   bash features/fetch_zenodo_kpms.sh --dry-run    # show what would be fetched
#   bash features/fetch_zenodo_kpms.sh --dest DIR   # override the destination
#
# Licence: CC-BY-4.0.  Cite Weinreb et al., Nature Methods 21:1329-1339 (2024)
# and the deposit DOI 10.5281/zenodo.15171335.  Nothing is redistributed here.

set -euo pipefail

RECORD=15171335
WANTED=open_field_2D.zip
API="https://zenodo.org/api/records/${RECORD}"

# Resolve the destination from the environment, paths.yml, or repository default.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

raw_root() {
  PYTHONPATH="$REPO" python3 -c 'from dlds_release.paths import RAW_ROOT; print(RAW_ROOT)'
}

DRY_RUN=0
DEST=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --dest) DEST="${2:?--dest needs a directory}"; shift ;;
    -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

command -v python3 >/dev/null || { echo "missing required tool: python3" >&2; exit 1; }
[ -n "$DEST" ] || DEST="$(raw_root)/keypoint_moseq_zenodo_15171335"

for tool in curl unzip; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done

md5_of() {  # md5sum on Linux, md5 on macOS
  if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1
  else md5 -q "$1"; fi
}

echo "record      : ${API}"
echo "destination : ${DEST}"

manifest="$(curl -fsSL --retry 3 "$API" | python3 -c '
import json, sys
rec = json.load(sys.stdin)
lic = (rec["metadata"].get("license") or {}).get("id", "unknown")
print("#license", lic, sep="\t")
for f in rec.get("files", []):
    print(f["key"], f["size"], f.get("checksum", ""), f["links"]["self"], sep="\t")
')"

echo "licence     : $(printf '%s\n' "$manifest" | awk -F'\t' '$1=="#license"{print $2}')"
echo
printf '%s\n' "$manifest" | awk -F'\t' '$1!~/^#/ {
  printf "  %-34s %12d  %s\n", $1, $2, ($1=="'"$WANTED"'" ? "TAKE" : "skip (unused by this paper)")
}'
echo

line="$(printf '%s\n' "$manifest" | awk -F'\t' -v k="$WANTED" '$1==k')"
[ -n "$line" ] || { echo "'$WANTED' is not in record $RECORD -- has the deposit changed?" >&2; exit 1; }

size=$(printf '%s\n' "$line" | cut -f2)
checksum=$(printf '%s\n' "$line" | cut -f3)   # "md5:<hex>"
url=$(printf '%s\n' "$line" | cut -f4)
want_md5="${checksum#md5:}"

if [ "$DRY_RUN" = 1 ]; then
  echo "DRY RUN. would download:"
  echo "  $url"
  echo "  -> $DEST/$WANTED   ($size bytes, md5 $want_md5)"
  echo "  then unzip into $DEST/ (the archive has open_field_2D/ at its top level)"
  exit 0
fi

mkdir -p "$DEST"
zip_path="$DEST/$WANTED"

# -C - resumes a partial download; large single-file transfers from Zenodo do
# get cut off, and a truncated zip is otherwise only noticed at unzip time.
curl -fL --retry 3 --retry-delay 5 -C - -o "$zip_path" "$url"

got_md5="$(md5_of "$zip_path")"
if [ "$got_md5" != "$want_md5" ]; then
  echo "MD5 mismatch for $zip_path" >&2
  echo "  expected $want_md5" >&2
  echo "  got      $got_md5" >&2
  echo "  the file is corrupt or truncated; delete it and re-run." >&2
  exit 1
fi
echo "md5 ok: $got_md5"

# -x '__MACOSX/*' rather than deleting afterwards: the deposit was zipped on a
# Mac and carries a parallel resource-fork tree that confuses the *.h5 glob.
unzip -q -o "$zip_path" -x '__MACOSX/*' -d "$DEST"

n_h5=$(find "$DEST/open_field_2D/videos" -name '*.h5' | wc -l)
n_mp4=$(find "$DEST/open_field_2D/videos" -name '*.mp4' | wc -l)
echo
echo "extracted to $DEST/open_field_2D"
echo "  config.yaml : $([ -f "$DEST/open_field_2D/config.yaml" ] && echo present || echo MISSING)"
echo "  DLC tracks  : $n_h5 (expected 10)"
echo "  videos      : $n_mp4 (expected 10; needed only for Fig-1a/1e)"
[ "$n_h5" = 10 ] || { echo "unexpected number of .h5 files" >&2; exit 1; }

cat <<'NEXT'

Next:
  python features/build_feature16_single_mouse.py
NEXT
