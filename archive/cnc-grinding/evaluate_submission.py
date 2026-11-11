# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
R_WHEEL_MM = 10.0


def point_to_polyline_distance(points: np.ndarray, poly: np.ndarray, chunk: int = 256) -> np.ndarray:
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    denom = np.sum(ab * ab, axis=1)
    denom = np.where(denom < 1e-18, 1e-18, denom)
    out = np.empty(len(points), dtype=float)
    for start in range(0, len(points), chunk):
        p = points[start : start + chunk]
        ap = p[:, None, :] - a[None, :, :]
        t = np.sum(ap * ab[None, :, :], axis=2) / denom[None, :]
        t = np.clip(t, 0.0, 1.0)
        proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        d2 = np.sum((p[:, None, :] - proj) ** 2, axis=2)
        out[start : start + chunk] = np.sqrt(np.min(d2, axis=1))
    return out


def normalize_submission_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "X_o": "x_center_mm",
        "Y_o": "y_center_mm",
        "x": "x_center_mm",
        "y": "y_center_mm",
        "X": "x_center_mm",
        "Y": "y_center_mm",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
    required = ["x_center_mm", "y_center_mm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "tool_id" not in df.columns:
        df.insert(0, "tool_id", np.arange(len(df), dtype=int))
    return df[["tool_id", "x_center_mm", "y_center_mm"]].copy()


def evaluate(submission: Path, output_dir: Path) -> pd.DataFrame:
    profile = pd.read_csv(HERE / "attachment2_profile_points_clean.csv", encoding="utf-8-sig")
    sub = pd.read_csv(submission, encoding="utf-8-sig")
    sub = normalize_submission_columns(sub)

    if len(sub) < 10:
        raise ValueError("Submission must contain at least 10 tool points.")
    if sub[["x_center_mm", "y_center_mm"]].isna().any().any():
        raise ValueError("Submission contains NaN in coordinate columns.")

    poly = profile[["x_mm", "y_mm"]].to_numpy(float)
    pts = sub[["x_center_mm", "y_center_mm"]].to_numpy(float)

    dist = point_to_polyline_distance(pts, poly)
    offset_err_um = (dist - R_WHEEL_MM) * 1000.0
    seg = np.hypot(np.roll(pts[:, 0], -1) - pts[:, 0], np.roll(pts[:, 1], -1) - pts[:, 1])
    mean_seg = float(seg.mean())
    seg_err_um = (seg - mean_seg) * 1000.0

    summary = pd.DataFrame(
        [
            {
                "point_count": len(sub),
                "mean_segment_mm": mean_seg,
                "max_equal_chord_error_um": float(np.max(np.abs(seg_err_um))),
                "rms_equal_chord_error_um": float(np.sqrt(np.mean(seg_err_um**2))),
                "max_abs_offset_error_um": float(np.max(np.abs(offset_err_um))),
                "rms_offset_error_um": float(np.sqrt(np.mean(offset_err_um**2))),
                "min_clearance_mm": float(dist.min()),
                "max_clearance_mm": float(dist.max()),
                "public_metric_note": "proxy_only_not_final_score",
            }
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    detail = sub.copy()
    detail["clearance_mm"] = dist
    detail["offset_error_um"] = offset_err_um
    detail.to_csv(output_dir / "evaluation_points.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "segment_id": np.arange(len(seg), dtype=int),
            "chord_length_mm": seg,
            "chord_error_um": seg_err_um,
        }
    ).to_csv(output_dir / "evaluation_segments.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "evaluation_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate B problem tool-center submission.")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_output"))
    args = parser.parse_args()
    summary = evaluate(args.submission, args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
