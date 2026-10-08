"""MoCap2MMD Desktop Application Entry Point."""
from __future__ import annotations

import os
import sys
import webview

from backend.api import MoCapAPI


def main():
    api = MoCapAPI()

    # Determine absolute path to frontend/index.html
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_path = os.path.join(base_dir, "frontend", "index.html")

    if not os.path.exists(frontend_path):
        print(f"Error: Frontend assets not found at {frontend_path}", file=sys.stderr)
        sys.exit(1)

    # Create PyWebView desktop window
    window = webview.create_window(
        title="MoCap2MMD - FBX / BVH to VMD Motion Retargeting Tool (MIT License)",
        url=frontend_path,
        js_api=api,
        width=1280,
        height=800,
        min_size=(960, 600),
        background_color="#0f1117",
    )
    api.set_window(window)

    # Start desktop GUI loop
    webview.start(debug=False)


if __name__ == "__main__":
    main()
