#!/usr/bin/env python3
"""Build the single-file LILA browser demo.

Takes web/index.html (which fetches lila.wasm + game.libyte) and injects
window.__LILA_EMBED__ = {wasm: <base64>, game: <base64>} before the module
script, producing ONE self-contained .html file that runs from anywhere
(file://, any static server, an email attachment) with zero other files.

Usage: python3 mkhtml.py [repo_root] [out_html] [game] [mode] [title]
  game   web/<game>.libyte  (default: game.libyte)
  mode   'shooter' | 'generic' — HUD/over-flag behavior of the v6 host
  title  replaces the page <title> suffix

Examples:
  mkhtml.py . ../build/lila-shooter.html game.libyte shooter
  mkhtml.py . ../build/lila-swarm.html  swarm.libyte generic "LILA ENGINE — Galaxy Swarm"
"""
import base64
import sys
from pathlib import Path

def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/z/my-project/lila-engine")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else root / "build" / "lila-shooter.html")
    game_name = sys.argv[3] if len(sys.argv) > 3 else "game.libyte"
    mode = sys.argv[4] if len(sys.argv) > 4 else "shooter"
    title = sys.argv[5] if len(sys.argv) > 5 else "LILA ENGINE — Wrap Shooter"

    html = (root / "web" / "index.html").read_text(encoding="utf-8")
    wasm = base64.b64encode((root / "web" / "lila.wasm").read_bytes()).decode("ascii")
    game = base64.b64encode((root / "web" / game_name).read_bytes()).decode("ascii")

    # classic <script> tag (async IIFE inside) — not type=module, so the file
    # also runs from file:// on iOS Safari, which can refuse module scripts
    marker = "<script>"
    if marker not in html:
        sys.exit("ERROR: script marker not found in web/index.html")

    # title + standalone suffix
    html = html.replace(
        "<title>LILA ENGINE — Wrap Shooter</title>",
        f"<title>{title} (standalone)</title>", 1)

    embed = (
        "<script>\n"
        f"window.__LILA_MODE__ = \"{mode}\";\n"
        f"window.__LILA_EMBED__ = {{ wasm: \"{wasm}\", game: \"{game}\" }};\n"
        "</script>\n"
    )
    html = html.replace(marker, embed + marker, 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"built {out} ({kb:.0f} KB, mode {mode}, wasm {len(wasm) * 3 // 4 // 1024} KB b64, "
          f"libyte {len(game) * 3 // 4} B b64)")

if __name__ == "__main__":
    main()
