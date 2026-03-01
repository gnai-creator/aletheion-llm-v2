"""
Aletheion LLM v2 - Tomografia Epistemica por Token

LLM com DRM/MAD/VI/MPC integrados como nn.Modules treinaveis.
Cada token produz coordenadas 5D no manifold epistemico,
incertezas Q1/Q2, confianca MAD e saude phi(M).
"""

__version__ = "0.1.0"
__author__ = "Felipe Maya Muniz"

from aletheion_v2.config import AletheionV2Config
from aletheion_v2.core.model import AletheionV2Model
from aletheion_v2.core.output import EpistemicTomography, ModelOutput

__all__ = [
    "AletheionV2Config",
    "AletheionV2Model",
    "EpistemicTomography",
    "ModelOutput",
]
