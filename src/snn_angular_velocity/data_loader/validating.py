import os

from .base import DatasetBase


class ValDatabase(DatasetBase):
    def __init__(self, data_dir: str):
        super().__init__(mode='val')
        assert os.path.isdir(data_dir)
        self.val_dir = os.path.join(data_dir, 'test')
        assert os.path.isdir(self.val_dir)

        self.subsequences = list()
        for dirpath, dirnames, filenames in os.walk(self.val_dir):
            assert not dirnames, 'This fails if the directory contains sub-directories. This should not happen'
            for filename in filenames:
                assert self.isDataFile(filename)
                self.subsequences.append(os.path.join(dirpath, filename))
