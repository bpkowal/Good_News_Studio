"""Serve only the saved-run viewer's static directory on loopback."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ViewerHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-cache')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'")
        super().end_headers()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=7860)
    p.add_argument('--refresh',action='store_true',help='Refresh the three bundled records from saved reports')
    a=p.parse_args()
    root=Path(__file__).resolve().parent/'frontend/dist'
    if a.refresh:
        from export_parliament_viewer import export
        export()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),partial(ViewerHandler,directory=str(root)))
    print(f'Parliament saved-run viewer: http://127.0.0.1:{server.server_port}',flush=True)
    print('Read-only. No API calls. Ctrl+C stops the server.',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
