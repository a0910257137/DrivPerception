from .base_attention import ChannelAttention, SelfAttention, PositionEmbeddingSine
from .conv_module import ConvBlock, TransitionUp, DepthwiseSeparableConv, TransposeUp
from .custom_losses import UncertaintyLoss, CoVWeightingLoss
from .layers import ASPP
from .base_attention import SelfAttention, ChannelAttention
from .common import MP, SPP, SPPF, SPPCSPC, Focus, BottleneckCSP, Bottleneck, RepConv, IDetect, SELayer

COMMON_FACTORY = dict(MP=MP,
                      SPPF=SPPF,
                      SPP=SPP,
                      SPPCSPC=SPPCSPC,
                      Focus=Focus,
                      Bottleneck=Bottleneck,
                      BottleneckCSP=BottleneckCSP,
                      RepConv=RepConv,
                      IDetect=IDetect,
                      SE=SELayer)
