from abc import ABC, abstractmethod

import tensorflow_datasets as tfds

from ml_prepare.datasets import datasets2classes
from ml_prepare.exectors import build_tfds_dataset


class BaseTrainer(ABC):
    """
    Trainer must be implemented for each ML framework
    """
    data = None

    @staticmethod
    def load_data_from_tfds_or_ml_prepare(dataset_name, tensorflow_datasets_dir=None, data_loader_kwargs=None):
        """

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param tensorflow_datasets_dir: directory to look for models in. Default is ~/tensorflow_datasets.
        :type tensorflow_datasets_dir: ```None or str```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```None or dict```

        :return: Train and tests dataset splits
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        if data_loader_kwargs is None:
            data_loader_kwargs = {}
        data_loader_kwargs.update({
            'dataset_name': dataset_name,
            'tfds_dir': tensorflow_datasets_dir
        })
        if dataset_name in datasets2classes:
            ds_builder = build_tfds_dataset(**data_loader_kwargs)
        else:
            ds_builder = BaseTrainer.get_from_tensorflow_datasets(dataset_name)

        if hasattr(ds_builder, 'download_and_prepare_kwargs'):
            download_and_prepare_kwargs = getattr(ds_builder, 'download_and_prepare_kwargs')
            delattr(ds_builder, 'download_and_prepare_kwargs')
        else:
            download_and_prepare_kwargs = None

        return BaseTrainer.common_dataset_handler(
            ds_builder=ds_builder,
            download_and_prepare_kwargs=download_and_prepare_kwargs,
            scale=None, K=None, as_numpy=False
        )

    @staticmethod
    def get_from_tensorflow_datasets(dataset_name, data_dir=None, K=None,
                                     as_numpy=False, scale=255., download_and_prepare_kwargs=None):
        """

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_dir: Where to look for the datasets, defaults to ~/tensorflow_datasets
        :type data_dir: ```None or str```

        :param K: backend engine, e.g., `np` or `tf`
        :type K: ```None or np or tf or Any```

        :param as_numpy: Convert to numpy ndarrays
        :type as_numpy: ```bool```

        :param scale: rescale input (divide) by this amount, None for do nothing
        :type scale: ```int or float or None```

        :param download_and_prepare_kwargs:
        :type download_and_prepare_kwargs: ```None or dict```

        :return: Train and tests dataset splits
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """

        return BaseTrainer.common_dataset_handler(
            ds_builder=tfds.builder(dataset_name, data_dir=data_dir),
            download_and_prepare_kwargs=download_and_prepare_kwargs,
            scale=scale, K=K, as_numpy=as_numpy
        )

    @staticmethod
    def common_dataset_handler(ds_builder, download_and_prepare_kwargs, scale, K, as_numpy):
        """

        :param ds_builder:
        :type ds_builder: ```tfds.core.DatasetBuilder```

        :param download_and_prepare_kwargs:
        :type download_and_prepare_kwargs: ```None or dict```

        :param scale: rescale input (divide) by this amount, None for do nothing
        :type scale: ```int or float or None```

        :param K: backend engine, e.g., `np` or `tf`
        :type K: ```None or np or tf or Any```

        :param as_numpy: Convert to numpy ndarrays
        :type as_numpy: ```bool```

        :return: Train and tests dataset splits
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        ds_builder.download_and_prepare(**(download_and_prepare_kwargs or {}))

        train_ds = ds_builder.as_dataset(split='train', batch_size=-1)
        test_ds = ds_builder.as_dataset(split='test', batch_size=-1)

        if as_numpy:
            train_ds, test_ds = map(tfds.as_numpy, (train_ds, test_ds))

        if K is not None and scale is not None:
            train_ds['image'] = K.float32(train_ds['image']) / scale
            test_ds['image'] = K.float32(test_ds['image']) / scale

        return train_ds, test_ds

    def load_data(self, dataset_name, data_loader=None,
                  data_loader_kwargs=None, data_type='infer',
                  output_type=None, K=None):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_loader: function that returns the expected data type.
         Defaults to TensorFlow Datasets and ml_prepare combined one.
        :type data_loader: ```None or (*args, **kwargs) -> tf.data.Datasets or Any```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```None or dict```

        :param data_type: incoming data type, defaults to 'infer'
        :type data_type: ```str```

        :param output_type: outgoing data_type, defaults to no conversion
        :type output_type: ```None or 'numpy'```

        :param K: backend engine, e.g., `np` or `tf`
        :type K: ```None or np or tf or Any```

        :return: Dataset splits (by default, your train and test)
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        assert dataset_name is not None
        if data_loader is None:
            data_loader = self.load_data_from_tfds_or_ml_prepare
        if data_type != 'infer':
            raise NotImplementedError(data_type)
        if data_loader_kwargs is None:
            data_loader_kwargs = {}
        data_loader_kwargs['dataset_name'] = dataset_name

        loaded_data = data_loader(**data_loader_kwargs)
        assert loaded_data is not None and hasattr(loaded_data, 'as_numpy')

        if output_type is None:
            self.data = loaded_data
        elif output_type == 'numpy' and hasattr(loaded_data, 'as_numpy'):
            self.data = loaded_data.as_numpy()
        else:
            raise NotImplementedError(output_type)

        return self.data

    @abstractmethod
    def train(self, epochs, *args, **kwargs):
        """

        :param epochs: number of epochs (must be greater than 0)
        :type epochs: int

        :param args:
        :param kwargs:
        :return:
        """
        """
        Run the training loop for your ML pipeline.
        """
        assert epochs is not None and epochs > 0
