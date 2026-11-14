#!/usr/bin/env python3
"""Preview container Vite on Mac localhost:5173 over verified SSH, without published ports."""
import argparse
import os
import selectors
import socket
import socketserver
import subprocess
import sys
import threading


def stdio_bridge():
    # Runs inside rex-coding. No private key or SSH server is needed there.
    with socket.create_connection(("127.0.0.1", 5173), timeout=10) as connection:
        connection.settimeout(None)
        with selectors.DefaultSelector() as selector:
            selector.register(sys.stdin.buffer, selectors.EVENT_READ)
            selector.register(connection, selectors.EVENT_READ)
            while True:
                for key, _ in selector.select():
                    if key.fileobj is connection:
                        data = connection.recv(65536)
                        if not data:
                            return
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                    else:
                        data = os.read(sys.stdin.fileno(), 65536)
                        if data:
                            connection.sendall(data)
                        else:
                            selector.unregister(sys.stdin.buffer)
                            connection.shutdown(socket.SHUT_WR)


def preview():
    processes = set()
    lock = threading.Lock()

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                       "rex-coding-sync", "~/.local/bin/coding python3 "
                       "/workspace/TinySettler/scripts/rex-preview.py --stdio"]
            with subprocess.Popen(command, stdin=self.request, stdout=self.request) as process:
                with lock:
                    processes.add(process)
                try:
                    process.wait()
                finally:
                    with lock:
                        processes.discard(process)

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    try:
        with Server(("127.0.0.1", 5173), Handler) as server:
            print("Remote preview: http://127.0.0.1:5173/catan.html (Ctrl-C to close)", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with lock:
            for process in processes:
                process.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdio", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    stdio_bridge() if args.stdio else preview()
