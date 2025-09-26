"""napari-socket plugin entry‐point"""
from ._widget import NapariSocketWidget

# npe2 needs the symbol available at module level
NapariSocketWidget = NapariSocketWidget