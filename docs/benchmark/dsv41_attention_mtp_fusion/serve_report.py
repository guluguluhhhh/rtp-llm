"""Read-only localhost report server. Ctrl-C stops it."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=8769)
a = p.parse_args()
root = Path(__file__).resolve().parent.parent
url = f"http://127.0.0.1:{a.port}/dsv41_attention_mtp_fusion/attention_mtp_fusion.html"
print(url, flush=True)
ThreadingHTTPServer(
    ("127.0.0.1", a.port), partial(SimpleHTTPRequestHandler, directory=str(root))
).serve_forever()
