"""Core: modelo base, embeddings, transformer blocks e output."""

from aletheion_v2.core.model import AletheionV2Model
from aletheion_v2.core.output import EpistemicTomography, ModelOutput

__all__ = ["AletheionV2Model", "EpistemicTomography", "ModelOutput"]
