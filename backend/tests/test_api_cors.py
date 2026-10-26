import asyncio
import unittest

from backend.main import app


class CorsOptionsTests(unittest.TestCase):
    def _call_options(self, origin: str) -> tuple[int, dict[str, str]]:
        scope = {
            "type": "http",
            "method": "OPTIONS",
            "path": "/api/routes/compare",
            "raw_path": b"/api/routes/compare",
            "query_string": b"",
            "headers": [
                (b"origin", origin.encode("utf-8")),
                (b"access-control-request-method", b"POST"),
                (b"access-control-request-headers", b"content-type"),
            ],
        }
        messages = []

        async def receive():
            return {"type": "http.request"}

        async def send(msg):
            messages.append(msg)

        asyncio.run(app(scope, receive, send))
        start_msg = next(m for m in messages if m["type"] == "http.response.start")
        headers = {k.decode("latin1"): v.decode("latin1") for k, v in start_msg.get("headers", [])}
        return start_msg["status"], headers

    def test_options_allows_localhost_5173(self) -> None:
        status, headers = self._call_options("http://localhost:5173")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("access-control-allow-origin"), "http://localhost:5173")

    def test_options_allows_127_0_0_1_5173(self) -> None:
        status, headers = self._call_options("http://127.0.0.1:5173")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("access-control-allow-origin"), "http://127.0.0.1:5173")

    def test_options_allows_alternate_vite_ports(self) -> None:
        status, headers = self._call_options("http://127.0.0.1:5174")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("access-control-allow-origin"), "http://127.0.0.1:5174")

        preview_status, preview_headers = self._call_options("http://localhost:4173")
        self.assertEqual(preview_status, 200)
        self.assertEqual(preview_headers.get("access-control-allow-origin"), "http://localhost:4173")

    def test_options_rejects_untrusted_origins(self) -> None:
        status, _ = self._call_options("http://untrusted-domain.com")
        self.assertEqual(status, 400)
