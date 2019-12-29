from pytext.models import Model
from pytext.config import ConfigBase
from pytext.data.tensorizers import TokenTensorizer
from pytext.models.representations.bilstm import BiLSTM
from pytext.models.embeddings.word_embedding import WordEmbedding
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.module import create_module
from pytext.models.output_layers import CRFOutputLayer
import torch

from typing import *


class MyTagger(Model):
    class Config(ConfigBase):
        class ModelInput(Model.Config.ModelInput):
            tokens: TokenTensorizer.Config = TokenTensorizer.Config(column='doc')
            slots: TokenTensorizer.Config = TokenTensorizer.Config(column='tags')

        inputs: ModelInput = ModelInput()
        embedding: WordEmbedding.Config = WordEmbedding.Config()
        representation: BiLSTM.Config = BiLSTM.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        output_layer: CRFOutputLayer.Config = CRFOutputLayer.Config()


    @classmethod
    def from_config(cls, config: Config, tensorizers):
        embedding: WordEmbedding = create_module(config.embedding, tensorizer=tensorizers['tokens'])
        representation: BiLSTM = create_module(config.representation, embed_dim=embedding.embedding_dim)
        slots = tensorizers['slots'].vocab
        decoder = create_module(config.decoder, in_dim=representation.representation_dim, out_dim=len(slots))
        output_layer = create_module(config.output_layer, labels=slots)
        return cls(embedding=embedding,
                   representation=representation,
                   decoder=decoder,
                   output_layer=output_layer)

    def arrange_model_inputs(self, tensor_dict):
        doc_tensor, length_tensor, _ = tensor_dict['tokens']
        return doc_tensor, length_tensor

    def arrange_targets(self, tensor_dict):
        slots, _, _ = tensor_dict['slots']
        return slots

    def arrange_model_context(self, tensor_dict):
        doc_tensor, length_tensor, _ = tensor_dict['tokens']
        return {'seq_lens': length_tensor}

    def forward(self, doc_tensor, length_tensor) -> List[torch.Tensor]:
        embedding = self.embedding(doc_tensor)
        rep = self.representation(embedding, length_tensor)

        if isinstance(rep, tuple):
            rep = rep[0]
        return self.decoder(rep)
