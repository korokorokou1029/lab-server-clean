"""Interpolate short gaps and smooth trajectory coordinates for selected locations."""

from pathlib import Path

import numpy as np
import pandas as pd


# 1. User settings
DATE_STR = "20260628"
TARGET_LOCATIONS = ["N2", "N3", "N4"]
MAX_INTERPOLATION_GAP = 30
WINDOW_SIZE = 31  # centered moving average, in frames
FILE_PATTERN = "homography_speed_angle_*.csv"

# Set this explicitly if automatic project-root discovery is unsuitable.
PROJECT_ROOT_OVERRIDE = None


def find_project_root() -> Path:
    """Find the project directory containing data/velocity&angle."""
    if PROJECT_ROOT_OVERRIDE is not None:
        return Path(PROJECT_ROOT_OVERRIDE).expanduser().resolve()

    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for directory in (start, *start.parents):
            if (directory / "data" / "velocity&angle").is_dir():
                return directory
    raise FileNotFoundError(
        "Project root not found. Set PROJECT_ROOT_OVERRIDE in the settings."
    )


def get_input_directory(project_root: Path, location: str) -> Path:
    """Prefer date/location folders; support the existing location/csv layout."""
    base = project_root / "data" / "velocity&angle"
    candidates = (
        base / DATE_STR / location / "csv",
        base / location / "csv",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Input CSV directory not found for {location}. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def process_csv(input_path: Path, output_directory: Path) -> dict:
    df = pd.read_csv(input_path)
    required_columns = ["frame_id", "track_id", "world_x", "world_y"]
    missing_columns = [column for column in required_columns if column not in df]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[["frame_id", "track_id"]].isna().any().any():
        raise ValueError("frame_id or track_id contains missing/non-numeric values")
    if not np.all(df["frame_id"] % 1 == 0):
        raise ValueError("frame_id contains non-integer values")
    df["frame_id"] = df["frame_id"].astype("int64")
    if df.duplicated(subset=["track_id", "frame_id"]).any():
        raise ValueError("Duplicate track_id/frame_id pairs found")

    output_columns = [
        "world_x_interp", "world_y_interp",
        f"world_x_ma{WINDOW_SIZE}", f"world_y_ma{WINDOW_SIZE}",
    ]
    for column in output_columns:
        df[column] = np.nan

    inserted_rows = []
    interpolated_frames = 0

    for track_id, track in df.groupby("track_id", sort=False):
        track = track.sort_values("frame_id")
        frames = track["frame_id"].to_numpy(dtype=np.int64)
        full_frames = np.arange(frames[0], frames[-1] + 1, dtype=np.int64)
        positions = frames - full_frames[0]

        xy = np.full((len(full_frames), 2), np.nan, dtype=float)
        xy[positions, 0] = track["world_x"].to_numpy(dtype=float)
        xy[positions, 1] = track["world_y"].to_numpy(dtype=float)
        valid = np.isfinite(xy).all(axis=1)
        xy[~valid] = np.nan

        interpolated = xy.copy()
        valid_positions = np.flatnonzero(valid)
        for left, right in zip(valid_positions[:-1], valid_positions[1:]):
            gap = right - left - 1
            if 1 <= gap <= MAX_INTERPOLATION_GAP:
                ratios = np.arange(1, gap + 1, dtype=float)[:, None] / (gap + 1)
                interpolated[left + 1:right] = (
                    interpolated[left]
                    + (interpolated[right] - interpolated[left]) * ratios
                )
                interpolated_frames += gap

        df.loc[track.index, "world_x_interp"] = interpolated[positions, 0]
        df.loc[track.index, "world_y_interp"] = interpolated[positions, 1]

        missing_row_mask = np.ones(len(full_frames), dtype=bool)
        missing_row_mask[positions] = False
        added_positions = np.flatnonzero(
            missing_row_mask & np.isfinite(interpolated).all(axis=1)
        )
        added_rows = None
        if len(added_positions):
            added_rows = pd.DataFrame(np.nan, index=range(len(added_positions)), columns=df.columns)
            added_rows["frame_id"] = full_frames[added_positions]
            added_rows["track_id"] = track_id
            if "class_id" in df.columns:
                previous = np.searchsorted(positions, added_positions, side="right") - 1
                added_rows["class_id"] = track["class_id"].to_numpy()[previous]
            added_rows["world_x_interp"] = interpolated[added_positions, 0]
            added_rows["world_y_interp"] = interpolated[added_positions, 1]

        smoothed = (
            pd.DataFrame(interpolated, columns=["x", "y"])
            .rolling(window=WINDOW_SIZE, center=True, min_periods=WINDOW_SIZE)
            .mean()
            .to_numpy()
        )
        df.loc[track.index, f"world_x_ma{WINDOW_SIZE}"] = smoothed[positions, 0]
        df.loc[track.index, f"world_y_ma{WINDOW_SIZE}"] = smoothed[positions, 1]
        if added_rows is not None:
            added_rows[f"world_x_ma{WINDOW_SIZE}"] = smoothed[added_positions, 0]
            added_rows[f"world_y_ma{WINDOW_SIZE}"] = smoothed[added_positions, 1]
            inserted_rows.append(added_rows)

    added_count = sum(len(rows) for rows in inserted_rows)
    if inserted_rows:
        df = pd.concat([df, *inserted_rows], ignore_index=True)
        df["frame_id"] = df["frame_id"].astype("int64")
        df = df.sort_values(["track_id", "frame_id"], kind="stable").reset_index(drop=True)

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{input_path.stem}_smoothed.csv"
    df.to_csv(output_path, index=False)

    valid_count = int(df[[f"world_x_ma{WINDOW_SIZE}", f"world_y_ma{WINDOW_SIZE}"]].notna().all(axis=1).sum())
    return {
        "file_name": input_path.name,
        "output_file": output_path.name,
        "total_rows": len(df),
        "added_rows": added_count,
        "interpolated_frames": interpolated_frames,
        f"ma{WINDOW_SIZE}_valid_rows": valid_count,
        f"ma{WINDOW_SIZE}_valid_ratio": valid_count / len(df) * 100 if len(df) else 0,
        "status": "ok",
    }


def process_location(project_root: Path, location: str) -> None:
    input_directory = get_input_directory(project_root, location)
    input_files = sorted(input_directory.glob(FILE_PATTERN))
    output_directory = (
        project_root / "output" / "cleaning" / DATE_STR / location / "moving_average"
    )
    if not input_files:
        print(f"[WARN] {location}: no matching CSV files in {input_directory}")
        return

    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"\n[LOCATION] {location}")
    print(f"[INPUT] {input_directory}")
    print(f"[OUTPUT] {output_directory}")
    results = []
    for index, input_path in enumerate(input_files, start=1):
        print(f"[{index}/{len(input_files)}] {input_path.name}")
        try:
            result = process_csv(input_path, output_directory)
            results.append(result)
            print(
                f"  done: added={result['added_rows']}, "
                f"interpolated={result['interpolated_frames']}, "
                f"MA{WINDOW_SIZE} valid={result[f'ma{WINDOW_SIZE}_valid_ratio']:.2f}%"
            )
        except Exception as exc:
            print(f"  error: {exc}")
            results.append({"file_name": input_path.name, "status": f"error: {exc}"})

    summary_path = output_directory / "processing_summary.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)
    succeeded = sum(result.get("status") == "ok" for result in results)
    print(f"[SUMMARY] {succeeded}/{len(input_files)} succeeded: {summary_path}")


def main() -> None:
    if WINDOW_SIZE % 2 != 1:
        raise ValueError("WINDOW_SIZE must be odd for a centered moving average")
    if MAX_INTERPOLATION_GAP < 0:
        raise ValueError("MAX_INTERPOLATION_GAP must be zero or greater")
    if not TARGET_LOCATIONS:
        raise ValueError("TARGET_LOCATIONS must contain at least one location")

    project_root = find_project_root()
    print(f"[BATCH] date={DATE_STR}, locations={TARGET_LOCATIONS}, root={project_root}")
    for location in TARGET_LOCATIONS:
        try:
            process_location(project_root, location)
        except Exception as exc:
            print(f"[ERROR] {location}: {exc}")


if __name__ == "__main__":
    main()
