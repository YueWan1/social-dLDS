"""
Per-frame REAL-video screenshots + a trimmed clip mp4 for the KP-MoSeq
single-mouse comparison clip, modeled on b3_worked_video_frames.py (CalMS21).

The open-field mp4 frame f is aligned 1:1 with the DLC keypoint frame f and with
the dLDS coefficient index (cs[:, f] is the transition x_f -> x_{f+1}; off by <=1,
negligible). We burn the FRAME NUMBER + TIME into the BOTTOM-LEFT corner of every
extracted frame so the video, the dLDS coefficient trace, and the KP-MoSeq syllable
sequence can all be aligned by the visible stamp.

As in the CalMS21 b3 clips (which drew both mice's MARS skeletons), we overlay the
mouse SKELETON on each frame: the 8 DLC keypoints used by dLDS/KP-MoSeq
(spine4..nose + both ears, tail dropped to match the analysis pipeline), connected
by the DLC config.yaml skeleton. Keypoints are taken from the DLC h5 in pixel
coordinates and cleaned with the same confidence interpolation as
kpmoseq_feature_pipeline.py (so the skeleton matches what the model saw).

Single-mouse data has NO behavior labels, so the stamp is just
"frame N / t = XX.XX s".

Clip (matches the dLDS coefficient panel in kpmoseq_joint_viz.ipynb):
  session 21_12_2_def6a_1, frames 75369-76269 (900 fr, 30 s @ 30 fps)

Outputs (under dLDS/results/kpms_repro/clip_video/<session>_fr<lo>_<hi>/):
  frames/fr<NNNNNN>.png          annotated raw frame (clean)
  frames_skel/fr<NNNNNN>.png     annotated frame with skeleton overlay
  clip_annotated.mp4             trimmed clip, stamp only
  clip_skeleton.mp4              trimmed clip, skeleton + stamp
"""
import os, sys, glob
import importlib.util
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw

from dlds_release.paths import RAW_ROOT, REPO, RESULTS_ROOT
from dlds_release.plotting import load_font

# The cleaning rule applied here MUST be the one the fit actually saw, so the
# skeleton overlay is derived from the feature builder itself rather than a
# copy.  That builder lives in the numbered stage directory features/, whose
# name is not a legal Python identifier, so it is loaded by path instead of
# imported.  (It was called kpmoseq_feature_pipeline.py in the original tree.)
_KPFP_PATH = REPO / 'features' / 'build_feature16_single_mouse.py'
_spec = importlib.util.spec_from_file_location('build_feature16_single_mouse', _KPFP_PATH)
kpfp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kpfp)               # reuse loader + outlier/interp

VID_DIR = RAW_ROOT / 'keypoint_moseq_zenodo_15171335' / 'open_field_2D' / 'videos'
OUT_ROOT = RESULTS_ROOT / 'kpms_repro' / 'clip_video'

FPS = 30.0
FONT = load_font(
    22,
    preferred=["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
)
STAMP_BORDER = (39, 119, 180)   # neutral blue box border

# (session_id, lo, hi)  -- the dLDS-vs-KP-MoSeq comparison clip
SESSION = '21_12_2_def6a_1'
LO, HI = 75369, 76269

# ── Skeleton: the 8 DLC keypoints used by dLDS/KP-MoSeq (tail dropped) ──
USE_PARTS = kpfp.USE_PARTS   # ['spine4','spine3','spine2','spine1','head','nose','right ear','left ear']
BONES = [('spine4', 'spine3'), ('spine3', 'spine2'), ('spine2', 'spine1'),
         ('spine1', 'head'), ('head', 'nose'),
         ('head', 'right ear'), ('head', 'left ear')]
# per-keypoint colors (RGB hex, matching the notebook KP_COL palette)
KP_RGB = {
    'spine4': (0x16, 0xa0, 0x85), 'spine3': (0x1a, 0xbc, 0x9c),
    'spine2': (0x27, 0xae, 0x60), 'spine1': (0xf1, 0xc4, 0x0f),
    'head':   (0xe6, 0x7e, 0x22), 'nose':   (0xe7, 0x4c, 0x3c),
    'right ear': (0xf3, 0x9c, 0x12), 'left ear': (0xd3, 0x54, 0x00),
}


def mp4_path(sid):
    return VID_DIR / f'{sid}.top.ir.mp4'


def load_clean_pixel_kp(sid):
    """Clean pixel-coord keypoints (T, 8, 2) for USE_PARTS, via the same load +
    outlier-zeroing + low-confidence linear interpolation as the feature pipeline
    (but WITHOUT the egocentric transform — we need raw image pixels for overlay)."""
    h5 = sorted(glob.glob(str(VID_DIR / f'{sid}.top.ir*500000.h5')))
    h5 = [p for p in h5 if os.path.basename(p).startswith(f'{sid}.top.ir')]
    assert len(h5) == 1, f'{sid}: {h5}'
    coords_raw, confs_raw = kpfp.load_dlc_h5(Path(h5[0]))        # (T,9,2),(T,9)
    confs_out, _ = kpfp.mark_outliers(coords_raw, confs_raw, kpfp.OUTLIER_SCALE)
    use = [kpfp.RAW_PARTS.index(p) for p in USE_PARTS]
    coords = coords_raw[:, use, :]
    confs = confs_out[:, use]
    coords = kpfp.interp_bad_frames(coords, confs, kpfp.CONF_THRESHOLD)
    return coords                                               # (T,8,2) pixels


def annotate(img_rgb, frame_idx):
    """Burn bottom-left 'frame N / t = XX.XX s' into an RGB image."""
    im = Image.fromarray(img_rgb)
    dr = ImageDraw.Draw(im)
    txt = f'frame {frame_idx}\nt = {frame_idx / FPS:6.2f} s'
    x0, pad = 8, 6
    bb = dr.multiline_textbbox((0, 0), txt, font=FONT, spacing=4)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    y0 = im.height - th - 2 * pad - 8
    dr.rectangle([x0, y0, x0 + tw + 2 * pad, y0 + th + 2 * pad],
                 fill=(255, 255, 255), outline=STAMP_BORDER, width=3)
    dr.multiline_text((x0 + pad, y0 + pad), txt, font=FONT,
                      fill=(34, 34, 34), spacing=4)
    return np.asarray(im)


def draw_skeleton_bgr(bgr, kp8):
    """Draw the 8-keypoint mouse skeleton onto a BGR frame in place.
    kp8: (8, 2) pixel coords in USE_PARTS order. Bones get a white outline + a
    thin colored line; keypoints are colored dots with a white edge (high contrast
    on the IR video)."""
    xy = {p: (int(round(kp8[i, 0])), int(round(kp8[i, 1])))
          for i, p in enumerate(USE_PARTS)}
    for a, b in BONES:
        cv2.line(bgr, xy[a], xy[b], (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(bgr, xy[a], xy[b], (180, 180, 180), 1, cv2.LINE_AA)
    for p in USE_PARTS:
        r, g, bl = KP_RGB[p]
        cv2.circle(bgr, xy[p], 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(bgr, xy[p], 2, (bl, g, r), -1, cv2.LINE_AA)   # BGR (half size)


def render_clip(sid, lo, hi, save_frames=True, clean=True, skeleton=True):
    out_dir = OUT_ROOT / f'{sid}_fr{lo}_{hi}'
    out_dir.mkdir(parents=True, exist_ok=True)
    fr_dir = out_dir / 'frames'
    sk_dir = out_dir / 'frames_skel'
    if save_frames and clean:
        fr_dir.mkdir(parents=True, exist_ok=True)
    if save_frames and skeleton:
        sk_dir.mkdir(parents=True, exist_ok=True)

    kp = load_clean_pixel_kp(sid) if skeleton else None

    cap = cv2.VideoCapture(str(mp4_path(sid)))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w_clean = cv2.VideoWriter(str(out_dir / 'clip_annotated.mp4'), fourcc, FPS, (W, H)) if clean else None
    w_skel = cv2.VideoWriter(str(out_dir / 'clip_skeleton.mp4'), fourcc, FPS, (W, H)) if skeleton else None

    n = 0
    for f in range(lo, hi):
        ok, bgr = cap.read()
        if not ok:
            print(f'  ! read failed at frame {f}')
            break
        if clean:
            rgb = annotate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), f)
            if save_frames:
                Image.fromarray(rgb).save(fr_dir / f'fr{f:06d}.png')
            if w_clean is not None:
                w_clean.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if skeleton:
            bgr_sk = bgr.copy()
            draw_skeleton_bgr(bgr_sk, kp[f])
            rgb_sk = annotate(cv2.cvtColor(bgr_sk, cv2.COLOR_BGR2RGB), f)
            if save_frames:
                Image.fromarray(rgb_sk).save(sk_dir / f'fr{f:06d}.png')
            if w_skel is not None:
                w_skel.write(cv2.cvtColor(rgb_sk, cv2.COLOR_RGB2BGR))
        n += 1
    cap.release()
    if w_clean is not None:
        w_clean.release()
    if w_skel is not None:
        w_skel.release()
    print(f'[{sid}] wrote {n} frames ({lo}-{hi}, {n / FPS:.1f} s) -> {out_dir}')
    if clean:
        print(f'  clean  : clip_annotated.mp4' + (f' + frames/' if save_frames else ''))
    if skeleton:
        print(f'  skeleton: clip_skeleton.mp4' + (f' + frames_skel/' if save_frames else ''))
    return out_dir


if __name__ == '__main__':
    # CLI: kpms_clip_video_frames.py SESSION LO HI
    sid = sys.argv[1] if len(sys.argv) > 1 else SESSION
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else LO
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else HI
    save_frames = os.environ.get('SKIP_FRAMES', '0') != '1'   # SKIP_FRAMES=1 -> mp4 only
    render_clip(sid, lo, hi, save_frames=save_frames)
