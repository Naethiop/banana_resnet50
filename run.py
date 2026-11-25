"""Start the application locally:  python run.py   then open http://127.0.0.1:5000 in a browser.

By default the server only accepts connections from this computer. To try it from a phone on the same Wi-Fi
network use --host 0.0.0.0 (other devices on that network can then reach it; use only on a network you trust).
"""
import argparse

from app.app import create_app

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    create_app().run(host=args.host, port=args.port, debug=False)
