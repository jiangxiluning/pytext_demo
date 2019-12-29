from pytext.data.sources.data_source import RootDataSource
from typing import *
from itertools import chain
from pathlib import Path
from random import Random
from torchtext.datasets.sequence_tagging import UDPOS
import torchtext
import torch


class TaggingDataSource(RootDataSource):

    class Config(RootDataSource.Config):
        folder: str = None
        fields: List[str] = ['text', 'slots']
        suffixes: List[str] = ['.txt']
        validation_split: float = 0.25
        random_seed = 1000

    @classmethod
    def from_config(cls, config: Config, schema: Dict[str, Type]):
        return cls(schema=schema, **config._asdict())

    def __init__(self, folder=Config.folder,
                 validation_split=Config.validation_split,
                 random_seed=Config.random_seed,
                 fields: List[str] = None,
                 suffixes: List[str] = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.data_dir = folder

        self.text_field = fields[0]
        self.slots_field = fields[1]

        self.suffixes = suffixes
        self.validation_split = validation_split
        self.random_seed = random_seed

    def _selector(self, select_eval):
        """
        This selector ensures that the same pseudo-random sequence is
        always the same from the beginning. The `select_eval` parameter
        guarantees that the training set and eval set are exact complements.
        """
        rng = Random(self.random_seed)

        def fn():
            return select_eval ^ (rng.random() >= self.validation_split)

        return fn

    def _iter_rows(self, select_fn=lambda: True):
        for doc, tags in TaggingDataSource.get_examples(Path(self.data_dir), self.suffixes):
            if select_fn():
                yield {self.text_field: doc, self.slots_field: tags}

    def raw_train_data_generator(self):
        return iter(
            self._iter_rows(select_fn=self._selector(select_eval=True))
        )

    def raw_eval_data_generator(self):
        return iter(
            self._iter_rows(select_fn=self._selector(select_eval=False))
        )

    @staticmethod
    def get_examples(folder, suffixes: list):
        if not folder.exists():
            raise RuntimeError('Folder is not existed.')

        for example in chain.from_iterable([folder.glob('*'.format(ext)) for ext in suffixes]):
            doc = []
            tags = []
            with example.open() as f:
                for line in f:
                    segs = line.strip().split('\t')
                    doc.append(segs[0])
                    tags.append(segs[1])
            yield doc, tags


class UDPOSDataSource(RootDataSource):

    class Config(RootDataSource.Config):
        fields: List[str] = ['text', 'slots']
        udpos_root: str = None

    @classmethod
    def from_config(cls, config: Config, schema: Dict[str, Type]):
        return cls(schema=schema, **config._asdict())

    def __init__(self, udpos_root=Config.udpos_root,
                 fields: List[str] = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.fields = fields

        text = torchtext.data.Field(sequential=True, use_vocab=True)
        slots = torchtext.data.Field(sequential=True, use_vocab=True)

        self.train_ds, self.val_ds, self.test_ds = UDPOS.splits(path=udpos_root,
                                                                root='.data',
                                                                fields=[
                                                                (fields[0], text),
                                                                (fields[1], slots),
                                                                ],
                                                                train='en-ud-tag.v2.train.txt',
                                                                validation='en-ud-tag.v2.dev.txt',
                                                                test='en-ud-tag.v2.test.txt')

    def _iter_row(self, ds: torchtext.data.Dataset):
        for example in ds:
            yield {self.fields[0]: getattr(example, self.fields[0]),
                   self.fields[1]: getattr(example, self.fields[1])}

    def raw_eval_data_generator(self):
        return iter(self._iter_row(self.val_ds))

    def raw_test_data_generator(self):
        return iter(self._iter_row(self.test_ds))

    def raw_train_data_generator(self):
        return iter(self._iter_row(self.train_ds))





if __name__ == '__main__':
    src = UDPOSDataSource('.data/udpos/en-ud-v2', fields=['text', 'slots'], schema={'text': List[str], 'slots': List[str]})
    for row in src.train:
        print(row)