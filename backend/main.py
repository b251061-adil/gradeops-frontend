from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pipeline import grade_exam

HOST = '0.0.0.0'
PORT = 8000

class GradeOpsHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()

    def do_GET(self):
        if self.path == '/health':
            self._set_headers()
            self.wfile.write(json.dumps({'status': 'ok', 'message': 'GradeOps backend is running.'}).encode('utf-8'))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({'error': 'Not found'}).encode('utf-8'))

    def do_POST(self):
        if self.path == '/grade':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode('utf-8') if length else '{}'
            try:
                payload = json.loads(body or '{}')
            except json.JSONDecodeError:
                self._set_headers(400)
                self.wfile.write(json.dumps({'error': 'Invalid JSON'}).encode('utf-8'))
                return

            result = grade_exam(payload)
            self._set_headers(200)
            self.wfile.write(json.dumps(result).encode('utf-8'))
            return

        if self.path == '/train':
            self._set_headers(200)
            self.wfile.write(json.dumps({'status': 'ok', 'message': 'No training needed for heuristic backend.'}).encode('utf-8'))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({'error': 'Not found'}).encode('utf-8'))


def run_server():
    server = HTTPServer((HOST, PORT), GradeOpsHandler)
    print(f'GradeOps backend running at http://{HOST}:{PORT}')
    server.serve_forever()


if __name__ == '__main__':
    run_server()
