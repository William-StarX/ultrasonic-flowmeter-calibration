from __future__ import annotations

import argparse
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "calibration"


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline calibration and evaluation of ultrasonic flowmeters.")
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate = commands.add_parser("calibrate", help="Fit a compensation model to reference measurements.")
    calibrate.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory with window, sample and geometry CSV files.")
    calibrate.add_argument("--output-dir", type=Path, default=Path("runs/calibration"))

    evaluate = commands.add_parser("evaluate", help="Compare window volume estimates against reference measurements.")
    evaluate.add_argument("--estimates", required=True, type=Path, help="CSV with window_id and model_volume_m3.")
    evaluate.add_argument("--reference", type=Path, default=DATA_DIR / "attachment1_window_data.csv")
    evaluate.add_argument("--output-dir", type=Path, default=Path("runs/evaluation"))

    args = parser.parse_args()
    try:
        if args.command == "calibrate":
            from final_postcalibrated_dual_kernel_krr import run

            run(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                report_dir=args.output_dir / "reports",
            )
        else:
            from official_evaluator import evaluate as evaluate_estimates

            evaluate_estimates(args.estimates, args.output_dir, args.reference)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
