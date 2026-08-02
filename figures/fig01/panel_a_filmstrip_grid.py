"""Reproduce the 2 x 5 locomotion filmstrip used in Figure 1a."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dlds_release.paths import RESULTS_ROOT, out_dir, require


VIDEO = RESULTS_ROOT / (
    "kpms_repro/clip_video/21_12_2_def6a_1_fr75369_76269/"
    "clip_skeleton.mp4"
)
CENTROIDS = Path(__file__).with_name("cents.npy")
OUTPUT = out_dir("fig01") / "filmstrip_locomotion_2x5.png"

START, STEP, N_FRAMES = 90, 24, 10
CROP_SIZE, SCALE = 175, 2
FRAME_SIZE = CROP_SIZE * SCALE
GUTTER = 6
FPS = 30


def interpolate_and_smooth(values: np.ndarray, width: int = 7) -> np.ndarray:
    result = values.astype(float)
    index = np.arange(len(result))
    good = np.isfinite(result)
    result[~good] = np.interp(index[~good], index[good], result[good])
    padded = np.pad(result, (width // 2, width // 2), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def read_selected_frames(path: Path) -> dict[int, np.ndarray]:
    wanted = {START + STEP * index for index in range(N_FRAMES)}
    selected = {}
    capture = cv2.VideoCapture(str(require(path)))
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in wanted:
            selected[frame_index] = frame
        frame_index += 1
    capture.release()
    missing = wanted - selected.keys()
    if missing:
        raise ValueError(f"Video ended before frames {sorted(missing)}")
    return selected


def main() -> None:
    centroids = np.load(require(CENTROIDS))
    xy = centroids[:, 1:3].astype(float)
    xy[centroids[:, 3] == 0] = np.nan
    center_x = interpolate_and_smooth(xy[:, 0])
    center_y = interpolate_and_smooth(xy[:, 1])
    frames = read_selected_frames(VIDEO)

    tiles = []
    for index in range(N_FRAMES):
        frame_number = START + STEP * index
        frame = frames[frame_number]
        x = int(round(center_x[frame_number]))
        y = int(round(center_y[frame_number]))
        x0 = np.clip(x - CROP_SIZE // 2, 0, frame.shape[1] - CROP_SIZE)
        y0 = np.clip(y - CROP_SIZE // 2, 0, frame.shape[0] - CROP_SIZE)
        crop = frame[y0 : y0 + CROP_SIZE, x0 : x0 + CROP_SIZE]
        crop = cv2.resize(crop, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_CUBIC)
        label = f"{frame_number / FPS * 1000:.0f} ms"
        cv2.putText(crop, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(crop, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("RGBA"))

    width = 5 * FRAME_SIZE + 4 * GUTTER
    height = 2 * FRAME_SIZE + GUTTER
    grid = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    for index, tile in enumerate(tiles):
        row, column = divmod(index, 5)
        grid.alpha_composite(
            tile,
            (column * (FRAME_SIZE + GUTTER), row * (FRAME_SIZE + GUTTER)),
        )
    grid.save(OUTPUT)
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
