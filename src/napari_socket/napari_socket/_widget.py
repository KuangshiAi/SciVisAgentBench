from qtpy.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton
from ._server import CommandServer


class NapariSocketWidget(QWidget):
    """Dock widget that starts/stops the server and shows the port."""
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        #self._srv: CommandServer | None = None
        from typing import Optional
        self._srv: Optional[CommandServer] = None

        self._lbl = QLabel("Server not running")
        btn_on  = QPushButton("Start server")
        btn_off = QPushButton("Stop server")

        btn_on.clicked.connect(self._start)
        btn_off.clicked.connect(self._stop)

        lay = QVBoxLayout(self)
        lay.addWidget(self._lbl); lay.addWidget(btn_on); lay.addWidget(btn_off)
        
        # Automatically start the server when widget is created
        self._start()

    # --------------------------------------------------------------------- #
    def _start(self):
        if self._srv is None:
            self._srv = CommandServer(port = 64908)
            self._srv.start()
            self._lbl.setText(f"Listening on 127.0.0.1:{self._srv.port}")
            
    def _stop(self):
        if self._srv:
            self._srv.shutdown()
            self._srv = None
            self._lbl.setText("Server stopped")