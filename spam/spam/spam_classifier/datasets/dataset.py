import os
from pathlib import Path
import shutil
from tempfile import mkdtemp
from typing import Tuple
from warnings import warn

import keras_preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.resnet_v2 import preprocess_input
import pandas as pd
from nsml.constants import DATASET_PATH

from keras.utils import Sequence
import imgaug.augmenters as iaa


class EmptyContentError(Exception):
    pass


UNLABELED = -1


class CustomInputGenerator(Sequence):
    """Wrapper of 2 ImageDataGenerator"""

    def __init__(self, fn, dataset, validation_fraction, batch_size, subset):
        # Keras generator
        self.generator = ImageDataGenerator(
            preprocessing_function=fn,
            horizontal_flip=True,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            validation_split=validation_fraction
        )
        self.dataset = dataset
        self.batch_size = batch_size
        self.subset = subset

        # Real time multiple input data augmentation
        self.genX1 = self.generator.flow_from_directory(
            directory=dataset.base_dir / 'train',
            shuffle=True,
            batch_size=self.batch_size,
            target_size=dataset.img_size[:-1],
            classes=dataset.classes,
            subset=subset)

        self.n = 0
        self.max = self.__len__()

    def __len__(self):
        """It is mandatory to implement it on Keras Sequence"""
        return self.genX1.__len__()

    def __getitem__(self, index):
        """Getting items from the 3 generators and packing them"""
        X_batch, Y_batch = self.genX1.__getitem__(index)

        mean = [0.55232704, 0.51815085, 0.48528248]
        std = [0.21313286, 0.21373375, 0.21965458]

        seq = iaa.Sequential([
            iaa.Multiply((0.1, 1.5)),
            iaa.contrast.LinearContrast((0.75, 1.5))
        ])

        X_batch = seq.augment_images(X_batch)
        for i in range(len(X_batch)):
            X_batch[i] = X_batch[i] / mean - std

        return X_batch, Y_batch

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result


class Dataset:
    """
    Basic dataset that can be used in combination with Keras fit_generator.
    Reorders the data to have one folder per class.
    """

    def __init__(self, classes, input_size):
        self.classes = classes
        self.img_size = input_size
        self.base_dir = Path(mkdtemp())
        self._len = None
        self.validation_fraction = 0.2

    def __del__(self):
        """
        Deletes the temporary folder that we created for the dataset.
        """
        shutil.rmtree(self.base_dir)

    def train_val_gen(self, batch_size: int):
        """
        Splits the train_data folder into train/val generators. Applies some image augmentation for the train dataset.

        Args:
            batch_size: int

        Returns:
            train_generator: Keras data generator.
            val_generator: Keras data generator.
        """
        train_generator = CustomInputGenerator(preprocess_input, self, self.validation_fraction, batch_size,
                                               'training')
        val_generator = CustomInputGenerator(preprocess_input, self, self.validation_fraction, batch_size,
                                             'validation')

        return train_generator, val_generator

    def test_gen(self, test_dir: str, batch_size: int):
        """
        Note that the test dataset is not rearranged.

        Args:
            test_dir: Path to the test dataseet.
            batch_size: Number of examples per batch. Reduce if encountering memory issues.

        Returns:
            gen: Keras generator for the test dataset.
            files: [str]
                A list of files. These are the same order as the images returned from the generator.

        """
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        files = [str(p.name) for p in (Path(test_dir) / 'test_data').glob('*.*') if p.suffix not in ['.gif', '.GIF']]
        metadata = pd.DataFrame({'filename': files})
        gen = datagen.flow_from_dataframe(metadata, directory=f'{test_dir}/test_data', x_col='filename',
                                          class_mode=None, shuffle=False, batch_size=batch_size)
        return gen, files

    def len(self, dataset):
        """
        Utility function to compute the number of datapoints in a given dataset.
        """
        if self._len is None:
            self._len = {
                dataset: sum([len(files) for r, d, files in os.walk(self.base_dir / dataset)]) for dataset in
                ['train']}
            self._len['train'] = int(self._len['train'] * (1 - self.validation_fraction))
            self._len['val'] = int(self._len['train'] * self.validation_fraction)
        return self._len[dataset]

    def prepare(self):
        """
        The resulting folder structure is compatible with the Keras function that generates a dataset from folders.
        """
        dataset = 'train'
        self._initialize_directory(dataset)
        self._rearrange(dataset)

    def _initialize_directory(self, dataset: str) -> None:
        """
        Initialized directory structure for a given dataset, in a way so that it's compatible with the Keras dataloader.
        """
        dataset_path = self.base_dir / dataset
        dataset_path.mkdir()
        for c in self.classes:
            (dataset_path / c).mkdir()

    def _rearrange(self, dataset: str) -> None:
        """
        Then rearranges the files based on the attached metadata. The resulting format is
        --
         |-train
             |-normal
                 |-img0
                 |-img1
                 ...
             |-monotone
                 ...
             |-screenshot
                 ...
             |_unknown
                 ...
        """
        output_dir = self.base_dir / dataset
        src_dir = Path(DATASET_PATH) / dataset
        metadata = pd.read_csv(src_dir / f'{dataset}_label')
        for _, row in metadata.iterrows():
            if row['annotation'] == UNLABELED:
                continue
            src = src_dir / 'train_data' / row['filename']
            if not src.exists():
                raise FileNotFoundError
            dst = output_dir / self.classes[row['annotation']] / row['filename']
            if dst.exists():
                warn(f'File {src} already exists, this should not happen. Please notify 서동필 or 방지환.')
            else:
                shutil.copy(src=src, dst=dst)
