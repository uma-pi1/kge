from kge.model.kge_model import KgeModel, KgeEmbedder
from kge.model.lookup_embedder import LookupEmbedder
from kge.model.projection_embedder import ProjectionEmbedder
from kge.model.complex import ComplEx
from kge.model.freex import Freex
from kge.model.distmult import DistMult
from kge.model.rescal import Rescal
from kge.model.sfnn import SFNN
from kge.model.sd_rescal import SparseDiagonalRescal
from kge.model.tucker3 import (
    Tucker3RelationEmbedder,
    SparseTucker3RelationEmbedder,
    RelationalTucker3,
)
from kge.model.fnn import Fnn
from kge.model.inverse_relations_model import InverseRelationsModel
from kge.model.conve import ConvE
from kge.model.transe import TransE
