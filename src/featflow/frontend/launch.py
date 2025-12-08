#!/usr/bin/env python3
"""Launch script for the frontend application."""

import sys
from pathlib import Path

project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now we can import normally
from featflow.frontend.app import main # noqa: E402

if __name__ == "__main__":
    main()
