from .slim_neck import SlimNeck
from .pafpn import PAFPN
from .pafpnx import PAFPNX

NECK_FACTORY = dict(pafpnx=PAFPNX, pafpn=PAFPN, slim_neck=SlimNeck)
