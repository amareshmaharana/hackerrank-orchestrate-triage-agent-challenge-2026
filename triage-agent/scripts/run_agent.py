"""Start interactive terminal agent session."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.terminal.cli import app

def main():
    app()

if __name__ == "__main__":
    main()
