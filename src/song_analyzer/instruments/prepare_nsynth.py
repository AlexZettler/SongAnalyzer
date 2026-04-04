"""CLI: download and prepare TensorFlow Datasets ``nsynth/full`` (TFRecords under TFDS data dir)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from song_analyzer.instruments.nsynth_logging import (
    configure_nsynth_logging,
    parse_train_log_level,
)
from song_analyzer.instruments.nsynth_train_loop import prepare_nsynth_tfrecords
from song_analyzer.instruments.train_nsynth import import_tfds_for_nsynth, resolve_tfds_data_dir


def prepare_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Prepare NSynth (nsynth/full) TFRecords via TensorFlow Datasets"
    )
    p.add_argument(
        "--tfds-data-dir",
        type=Path,
        default=None,
        help="TensorFlow Datasets root (default: TFDS_DATA_DIR env or ~/tensorflow_datasets)",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
        help="Python logging level for messages",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Append logs to this file (default: platform log dir / nsynth_prepare.log)",
    )
    p.add_argument(
        "--no-log-file",
        action="store_true",
        help="Do not write a log file (console only)",
    )
    p.add_argument(
        "--log-rss-interval-seconds",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "During download_and_prepare, log process RSS every SEC seconds "
            "(requires psutil from the [train] extra)"
        ),
    )
    args = p.parse_args(argv)

    configure_nsynth_logging(
        parse_train_log_level(args.log_level),
        profile="prepare",
        log_file=args.log_file,
        no_log_file=args.no_log_file,
    )
    _, tfds = import_tfds_for_nsynth()
    data_dir = resolve_tfds_data_dir(args.tfds_data_dir)
    out = prepare_nsynth_tfrecords(
        tfds,
        data_dir=data_dir,
        log_rss_interval_seconds=args.log_rss_interval_seconds,
    )
    print(f"NSynth TFDS variant directory (TFRecords): {out}", file=sys.stdout)


if __name__ == "__main__":
    prepare_main()
