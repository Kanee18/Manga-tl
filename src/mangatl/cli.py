"""Command line access to the same pipeline the window drives.

Useful for translating a chapter without a display attached, and for checking
that the models and Ollama are reachable when the GUI reports it cannot start.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import default_config, load_config
from .runner import ChapterRunner, find_pages, output_dir_for
from .runtime import describe_device


def _bar(fraction: float, width: int = 28) -> str:
    filled = int(fraction * width)
    return "#" * filled + "-" * (width - filled)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mangatl",
        description="Translate English manga into Indonesian, offline.",
    )
    parser.add_argument("chapter", type=Path, nargs="?", help="folder of page images")
    parser.add_argument("-o", "--out", type=Path, help="where to write the result")
    parser.add_argument(
        "-p", "--page", type=int, action="append",
        help="translate only this page number (1-based); repeatable",
    )
    parser.add_argument("-c", "--config", type=Path, help="use a different config.yaml")
    parser.add_argument("--model", help="override the Ollama model")
    parser.add_argument("--debug", action="store_true", help="write stage overlays")
    parser.add_argument(
        "--check", action="store_true", help="report readiness and exit"
    )
    parser.add_argument("--gui", action="store_true", help="open the desktop app")
    args = parser.parse_args(argv)

    if args.gui:
        from .gui import main as gui_main

        return gui_main([sys.argv[0]] + ([str(args.chapter)] if args.chapter else []))

    cfg = load_config(args.config) if args.config else default_config()
    if args.model:
        cfg = cfg.model_copy(deep=True)
        cfg.translate.model = args.model
    if args.debug:
        cfg = cfg.model_copy(deep=True)
        cfg.output.debug = True

    print(f"device: {describe_device()}")
    runner = ChapterRunner(cfg=cfg)
    try:
        ok, message = runner.check_ready()
        print(f"translator: {'ok' if ok else 'NOT READY'} -- {message}")
        if args.check:
            return 0 if ok else 1
        if not ok:
            return 1

        if args.chapter is None:
            parser.error("a chapter folder is required (or pass --check)")

        pages = find_pages(args.chapter)
        if not pages:
            print(f"no page images in {args.chapter}", file=sys.stderr)
            return 1
        if args.page:
            wanted = {n - 1 for n in args.page}
            pages = [p for i, p in enumerate(pages) if i in wanted]
            if not pages:
                print("no pages matched --page", file=sys.stderr)
                return 1

        out_dir = args.out or output_dir_for(args.chapter, cfg)
        print(f"{len(pages)} page(s) -> {out_dir}\n")

        failures = 0
        for index, path in enumerate(pages):
            def progress(label: str, fraction: float, i: int = index) -> None:
                print(
                    f"\r  [{_bar(fraction)}] {i + 1}/{len(pages)} {label:<32}",
                    end="", flush=True,
                )

            result = runner.translate_page(path, out_dir, progress)
            print("\r" + " " * 78, end="\r")
            skipped = f", {result.skipped} left as art" if result.skipped else ""
            print(f"  {path.name}: {result.typeset}/{result.attempted} lettered"
                  f"{skipped} in {result.seconds:.1f}s")
            for problem in result.problems:
                failures += 1
                print(f"      {problem}")

        print(f"\nDone. {len(pages)} page(s), {failures} region(s) needing attention.")
        return 0
    finally:
        runner.close()


if __name__ == "__main__":
    sys.exit(main())
