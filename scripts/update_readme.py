from __future__ import annotations
from pathlib import Path
import sys

READme = Path("README.md")
LB = Path("leaderboard.md")

START = "<!-- LB-START -->"
END = "<!-- LB-END -->"

def main() -> None:
    if not LB.exists():
        print("leaderboard.md not found. Run the leaderboard generator first.", file=sys.stderr)
        sys.exit(1)

    readme_text = READme.read_text(encoding="utf-8")
    lb_text = LB.read_text(encoding="utf-8").strip()

    if START not in readme_text or END not in readme_text:
        print("Markers not found in README.md", file=sys.stderr)
        sys.exit(1)

    before, rest = readme_text.split(START, 1)
    _, after = rest.split(END, 1)
    new_block = f"{START}\n\n{lb_text}\n\n{END}"
    updated = before + new_block + after

    READme.write_text(updated, encoding="utf-8")
    print("README.md updated with leaderboard.")

if __name__ == "__main__":
    main()
