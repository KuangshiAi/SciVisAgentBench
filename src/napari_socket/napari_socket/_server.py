#import json, socketserver, threading
#from napari._qt.qt_main_window import Window
# from napari.utils import get_app
import json, socketserver, threading, queue
from qtpy.QtCore import QObject, Signal, Qt
from napari._app_model import get_app_model

# marshal commands to the GUI thread ----------------------------------
class _Dispatcher(QObject):
    # include a Queue argument that will receive the return-value
    exec_cmd = Signal(str, list, object)

# create once, on the main thread (module import happens in GUI thread)
_dispatcher = _Dispatcher()
# slot: run command on GUI thread and push *result* into the queue
_dispatcher.exec_cmd.connect(
    lambda cid, a, q: q.put(get_app_model().commands.execute_command(cid, *a)),
    Qt.QueuedConnection,
)

class _TCPHandler(socketserver.BaseRequestHandler):
    """
    One handler per incoming connection.
    Expects a single JSON line: ["command.id", [arg1, arg2, ...]]
    """
    def handle(self):
        data = self.request.recv(8192).decode().strip()
        try:
            cmd_id, args = json.loads(data)
            print(threading.current_thread())
            # one queue per request
            resp_q: queue.Queue = queue.Queue()
            # execute on GUI thread (queued)
            _dispatcher.exec_cmd.emit(cmd_id, args or [], resp_q)
            result = resp_q.get()            # ← blocks until done

            # If result is a Future, get its result
            if hasattr(result, "result") and callable(result.result):
                try:
                    result = result.result(timeout=20)
                except Exception as e:
                    self.request.sendall(f"ERR {e}\n".encode())
                    return

            try:
                payload = json.dumps(result)
                reply: bytes = f"OK {payload}\n".encode()
            except TypeError:                # result not JSON-serialisable
                reply = b"OK\n"

            self.request.sendall(reply)
        except Exception as exc:
            self.request.sendall(f"ERR {exc}\n".encode())

class CommandServer(threading.Thread):
    """
    Runs `socketserver.TCPServer` in its own thread so Qt stays responsive.
    """
    #def __init__(self, host="127.0.0.1", port=0):
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        super().__init__(daemon=True)
        self._srv = socketserver.TCPServer((host, port), _TCPHandler, bind_and_activate=False)
        self._srv.allow_reuse_address = True
        self._srv.server_bind()
        self._srv.server_activate()

    # public -----------------------------------------------------------------
    @property
    def port(self) -> int:
        return self._srv.server_address[1]

    def run(self):
        self._srv.serve_forever()

    def shutdown(self):
        self._srv.shutdown()