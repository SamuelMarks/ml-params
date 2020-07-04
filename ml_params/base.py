from abc import ABC, abstractmethod
from typing import Tuple, Any

import numpy as np

from ml_params.datasets import load_data_from_tfds_or_ml_prepare

try:
    import tensorflow as tf
except ImportError:
    tf = None  # `tf` is only used for typings in this file

from ml_prepare.exectors import build_tfds_dataset


class BaseTrainer(ABC):
    """
    Trainer that is be implemented for each ML framework
    """
    data = None  # type: (None or Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray])
    model = None  # type: (None or Any or tf.keras.models.Sequential or tf.keras.models.Model)

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
            data_loader = load_data_from_tfds_or_ml_prepare
        if data_type != 'infer':
            raise NotImplementedError(data_type)
        if data_loader_kwargs is None:
            data_loader_kwargs = {}
        data_loader_kwargs['dataset_name'] = dataset_name
        if 'data_loader_kwargs' not in data_loader_kwargs:
            data_loader_kwargs['data_loader_kwargs'] = {}
        data_loader_kwargs['data_loader_kwargs']['K'] = K
        if 'as_numpy' in data_loader_kwargs:
            data_loader_kwargs['data_loader_kwargs']['as_numpy'] = data_loader_kwargs.pop('as_numpy')

        loaded_data = data_loader(**data_loader_kwargs)
        assert loaded_data is not None

        if output_type is None:
            self.data = loaded_data
        elif output_type == 'numpy':
            if hasattr(loaded_data, 'as_numpy'):
                self.data = loaded_data.as_numpy()
        else:
            raise NotImplementedError(output_type)

        return self.data

    @abstractmethod
    def load_model(self, model, model_kwargs=None, call=False):
        """

        :param model: model object, e.g., a tf.keras.Sequential, tl.Serial,  nn.Module instance

        :param model_kwargs: to be passed into the model. If empty, doesn't call, unless call=True.
        :type model_kwargs: ```None or dict```

        :param call: call `model()` even if `model_kwargs is None`
        :type call: ```bool```

        :return self.model, e.g., the result of applying `model_kwargs` on model
        """

        self.model = model if model_kwargs is None and not call else model(**(model_kwargs or {}))
        return self.model

    @abstractmethod
    def train(self, epochs, save_directory, *args, **kwargs):
        """
        Run the training loop for your ML pipeline.

        :param epochs: number of epochs (must be greater than 0)
        :type epochs: ```int```

        :param save_directory: directory to save output to, e.g., weights
        :type save_directory: ```bool```

        :param args:
        :param kwargs:
        :return:
        """
        assert epochs is not None and epochs > 0


del ABC, abstractmethod, np, tf, build_tfds_dataset

__all__ = ['BaseTrainer']
