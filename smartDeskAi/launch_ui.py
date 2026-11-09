"""
launch_ui.py — `moxa-ui` console script: launch the Streamlit app.

Kept separate from app.py so launching doesn't import the app (and run its Streamlit cache
decorators) before the Streamlit runtime exists.
"""

import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "app.py"


def run() -> None:
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(APP_PATH), *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    run()
