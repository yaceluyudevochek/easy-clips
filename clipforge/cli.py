"""Command line: `python -m clipforge run VIDEO_OR_URL` and `python -m clipforge doctor`."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config


def cmd_run(args, cfg) -> int:
    from .pipeline import STAGES, run_pipeline

    force = set(args.force or [])
    if "all" in force:
        force = set(STAGES)
    run_pipeline(args.source, cfg, max_clips=args.max_clips, force=force)
    return 0


def cmd_doctor(args, cfg) -> int:
    from .doctor import run_checks

    checks = run_checks(cfg)
    for c in checks:
        print(f"[{'OK' if c.ok else 'FAIL'}] {c.name}" + (f"\n       -> {c.hint}" if c.hint else ""))
    ok = all(c.ok for c in checks)
    print("\nВсё готово!" if ok else "\nИсправьте пункты FAIL выше.")
    return 0 if ok else 1


def cmd_gui(args, cfg) -> int:
    from .gui import run_gui

    return run_gui(args.config)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="clipforge", description="Cut long videos into vertical viral clips")
    p.add_argument("-c", "--config", type=Path, help="YAML config (default: ./config.yaml if present)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="process a video file or URL")
    r.add_argument("source", help="path to a video or a YouTube/other URL")
    r.add_argument("-n", "--max-clips", type=int, help="override highlights.max_clips")
    r.add_argument("--force", nargs="*", choices=["transcribe", "highlights", "render", "all"],
                   help="redo stages even if cached results exist")
    sub.add_parser("doctor", help="check the environment (ffmpeg, NVENC, CUDA, LLM)")
    sub.add_parser("gui", help="open the desktop app")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    handler = {"run": cmd_run, "doctor": cmd_doctor, "gui": cmd_gui}[args.command]
    try:
        return handler(args, cfg)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        if args.verbose:
            raise
        logging.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
