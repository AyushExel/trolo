from trolo.loaders import register
from ..dfine.dfine import DFINE

@register()
class RTDETR(DFINE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

