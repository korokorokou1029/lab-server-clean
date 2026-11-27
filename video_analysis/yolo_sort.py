"""Multiple-location YOLO/ByteTrack batch analysis. Run with nohup on Linux."""
from pathlib import Path
import csv
import json
import time
import traceback
import cv2
import numpy as np
from ultralytics import YOLO

# 1. User settings
DATE_STR = '20260628'
TARGET_LOCATIONS = ['N2', 'N3', 'N4']
DEVICE = 'cpu'  # この実行環境ではCUDA/MPSが利用できないためCPUを使用
CONF_THRESHOLD = 0.25
TARGET_CLASSES = list(range(10))
TRACKER = 'bytetrack.yaml'
PROGRESS_INTERVAL = 200
# Set explicitly if automatic project root discovery is unsuitable.
PROJECT_ROOT_OVERRIDE = None

# 2. Paths

def find_project_root():
    if PROJECT_ROOT_OVERRIDE is not None:
        return Path(PROJECT_ROOT_OVERRIDE).expanduser().resolve()
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for path in (start, *start.parents):
            if (path / 'raw_video').is_dir() and (path / 'weights').is_dir():
                return path
    raise RuntimeError('Project root not found. Set PROJECT_ROOT_OVERRIDE.')


def load_coordinates(path):
    coordinates = {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            location = row[reader.fieldnames[0]].strip()
            if location not in TARGET_LOCATIONS:
                continue
            try:
                points = np.array([list(map(int, row[c].split(',')))
                                   for c in ('左下', '右下', '右上', '左上')])
                if points.shape != (4, 2):
                    raise ValueError('Expected four (x, y) points')
                coordinates[location] = points
            except (ValueError, KeyError, AttributeError) as exc:
                print(f'[WARN] {location}: invalid or missing coordinates: {exc}', flush=True)
    return coordinates


def crop_bounds(points):
    x0, y0 = map(int, points.min(axis=0))
    x1, y1 = map(int, points.max(axis=0))
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError('Invalid crop bounds')
    # Ensure encoded video and normalized labels use the same even dimensions.
    x1 -= (x1 - x0) % 2
    y1 -= (y1 - y0) % 2
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Crop too small')
    return x0, y0, x1, y1


def process_video(path, output_root, bounds, weights):
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0, y1 - y0
    dest = output_root / path.parent.name / path.stem
    labels = dest / 'labels'
    video = dest / f'{path.stem}.mp4'
    flag = dest / 'completed.flag'
    signature = dict(source=str(path.resolve()), source_size=path.stat().st_size,
                     source_mtime_ns=path.stat().st_mtime_ns, bounds=list(bounds),
                     weights=str(weights), weights_mtime_ns=weights.stat().st_mtime_ns,
                     confidence=CONF_THRESHOLD, classes=TARGET_CLASSES, tracker=TRACKER)
    if flag.is_file() and video.is_file() and video.stat().st_size > 0 and labels.is_dir():
        try:
            if json.loads(flag.read_text(encoding='utf-8'))['settings'] == signature:
                print(f'[SKIP] {path.name}', flush=True)
                return 'skipped'
        except (ValueError, KeyError, OSError):
            pass
    cap = cv2.VideoCapture(str(path))
    out = None
    count = 0
    start = time.monotonic()
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open input video')
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not np.isfinite(fps) or fps <= 0 or x1 > ow or y1 > oh:
            raise ValueError(f'Invalid FPS or crop outside {ow}x{oh}')
        # A fresh model per video ensures tracker IDs/state do not carry over.
        model = YOLO(str(weights))
        labels.mkdir(parents=True, exist_ok=True)
        flag.unlink(missing_ok=True)
        for old in labels.glob('*.txt'):
            old.unlink()
        video.unlink(missing_ok=True)
        out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not out.isOpened():
            raise RuntimeError('Cannot open output video writer')
        print(f'[START] {path.name}: {total} frames, {fps:.2f} FPS, crop {width}x{height}', flush=True)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            cropped = frame[y0:y1, x0:x1]
            if cropped.shape[:2] != (height, width):
                raise RuntimeError('Unexpected crop dimensions')
            result = model.track(source=cropped, persist=True, classes=TARGET_CLASSES,
                                 conf=CONF_THRESHOLD, verbose=False, device=DEVICE,
                                 tracker=TRACKER)[0]
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                lines = []
                for box, tid, cls, conf in zip(boxes.xywhn.cpu().numpy(),
                                               boxes.id.cpu().numpy().astype(int),
                                               boxes.cls.cpu().numpy().astype(int),
                                               boxes.conf.cpu().numpy()):
                    xc, yc, w, h = box
                    lines.append(f'{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.4f} {tid}\n')
                if lines:
                    (labels / f'{path.stem}_{count}.txt').write_text(''.join(lines), encoding='utf-8')
            out.write(result.plot())
            if count % PROGRESS_INTERVAL == 0 or count == total:
                pct = 100 * count / total if total > 0 else 0
                rate = count / max(time.monotonic() - start, 1e-9)
                print(f'[{path.parent.name}/{path.name}] {count}/{total} ({pct:.1f}%) | {rate:.1f} fps', flush=True)
        if count == 0 or (total > 0 and count != total):
            raise RuntimeError(f'Incomplete input: {count}/{total} frames')
    finally:
        cap.release()
        if out is not None:
            out.release()
    check = cv2.VideoCapture(str(video))
    try:
        if (not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != count
                or int(check.get(cv2.CAP_PROP_FRAME_WIDTH)) != width
                or int(check.get(cv2.CAP_PROP_FRAME_HEIGHT)) != height):
            raise RuntimeError('Output video validation failed')
    finally:
        check.release()
    temp = dest / 'completed.tmp'
    temp.write_text(json.dumps({'settings': signature, 'processed_frames': count}, indent=2), encoding='utf-8')
    temp.replace(flag)
    print(f'[DONE] {path.name}: {(time.monotonic()-start)/60:.2f} min', flush=True)
    return 'completed'


def main():
    root = find_project_root()
    output = root / 'output' / 'yoloresults_cropped'
    weights = root / 'weights' / 'yolov8_basemodel_20240131_v3.pt'
    if not weights.is_file():
        raise FileNotFoundError(weights)
    coordinates = load_coordinates(output / 'coordinate_for_yolo.csv')
    counts = dict(completed=0, skipped=0, failed=0)
    unavailable = []
    print(f'[BATCH] {DATE_STR}, {TARGET_LOCATIONS}, device={DEVICE}, root={root}', flush=True)
    for i, location in enumerate(TARGET_LOCATIONS, 1):
        print(f'\n[LOCATION {i}/{len(TARGET_LOCATIONS)}] {location}', flush=True)
        source = root / 'raw_video' / DATE_STR / f'{DATE_STR} {location}'
        try:
            if location not in coordinates:
                raise ValueError('Coordinates missing')
            bounds = crop_bounds(coordinates[location])
            videos = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == '.mp4')
            if not videos:
                raise ValueError('No MP4 files')
        except (ValueError, OSError) as exc:
            unavailable.append(location)
            print(f'[WARN] {location}: {exc}', flush=True)
            continue
        for j, path in enumerate(videos, 1):
            print(f'[VIDEO {j}/{len(videos)}] {path.name}', flush=True)
            try:
                status = process_video(path, output, bounds, weights)
            except Exception:
                traceback.print_exc()
                status = 'failed'
            counts[status] += 1
    print(f'[SUMMARY] {counts}, unavailable locations: {unavailable}', flush=True)
    return 1 if counts['failed'] or unavailable else 0


if __name__ == '__main__':
    raise SystemExit(main())
