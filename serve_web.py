"""
Nepal Sewa AI — Simple HTTP Server
Serves the web UI and proxies API requests to the FastAPI backend.

Usage:
    python serve_web.py [--port 3000] [--backend-port 8000]
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import threading
import urllib.request
import urllib.error
from pathlib import Path

# Path to the HTML file
HTML_FILE = Path(__file__).parent / "nepal_sewa_ai.html"


class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the HTML file and proxies /api/* to the FastAPI backend."""

    backend_port: int = 8000

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._proxy_request("GET")
        elif self.path == "/" or self.path == "/index.html":
            self._serve_html()
        else:
            self._serve_html()  # SPA fallback

    def do_POST(self):
        if self.path.startswith("/api/"):
            self._proxy_request("POST")
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serve_html(self):
        """Serve the main HTML file."""
        if not HTML_FILE.exists():
            self.send_error(404, "HTML file not found")
            return
        content = HTML_FILE.read_text(encoding="utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def _proxy_request(self, method: str):
        """Proxy API request to FastAPI backend."""
        url = f"http://127.0.0.1:{self.backend_port}{self.path}"

        # Read request body for POST
        body = None
        if method == "POST":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else None

        try:
            req = urllib.request.Request(url, data=body, method=method)
            req.add_header("Content-Type", "application/json")

            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)

        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            error = json.dumps({"error": f"Backend unavailable: {str(e)}"})
            self.wfile.write(error.encode("utf-8"))

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Handle requests in separate threads."""
    daemon_threads = True
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description="Nepal Sewa AI Web Server")
    parser.add_argument("--port", type=int, default=3000, help="Web server port")
    parser.add_argument("--backend-port", type=int, default=8000, help="FastAPI backend port")
    args = parser.parse_args()

    ProxyHandler.backend_port = args.backend_port

    server = ThreadedServer(("127.0.0.1", args.port), ProxyHandler)
    print(f"\n{'='*60}")
    print(f"  🇳🇵  Nepal Sewa AI — Web Interface")
    print(f"{'='*60}")
    print(f"  Web UI:    http://localhost:{args.port}")
    print(f"  Backend:   http://localhost:{args.backend_port}")
    print(f"{'='*60}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
