from abc import ABC, abstractmethod
from sys import stdout
from typing import Tuple, Any, List

from ml_params.datasets import load_data_from_tfds_or_ml_prepare
from ml_params.utils import to_numpy

try:
    import tensorflow as tf
    import numpy as np
except ImportError:
    tf = None  # `tf` is only used for typings in this file
    np = None  # `np` is only used for typings in this file


class BaseTrainer(ABC):
    """
    Trainer that is to be implemented for each ML framework
    """
    data = None  # type: (None or Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray])
    model = None  # type: (None or Any or tf.keras.models.Sequential or tf.keras.models.Model)

    def load_data(self, dataset_name, data_loader=load_data_from_tfds_or_ml_prepare,
                  data_type='infer', output_type=None, K=None,
                  **data_loader_kwargs):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_loader: function that returns the expected data type.
         Defaults to TensorFlow Datasets and ml_prepare combined one.
        :type data_loader: ```(*args, **kwargs) -> tf.data.Datasets or Any```

        :param data_type: incoming data type, defaults to 'infer'
        :type data_type: ```str```

        :param output_type: outgoing data_type, defaults to no conversion
        :type output_type: ```None or 'numpy'```

        :param K: backend engine, e.g., `np` or `tf`
        :type K: ```None or np or tf or Any```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```**data_loader_kwargs```

        :return: Dataset splits (by default, your train and test)
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        assert dataset_name is not None
        if data_type != 'infer':
            raise NotImplementedError(data_type)
        elif data_loader is None:
            data_loader = load_data_from_tfds_or_ml_prepare

        data_loader_kwargs.update({
            'dataset_name': dataset_name,
            'K': K,
            'as_numpy': data_loader_kwargs.get('as_numpy', True)
        })

        loaded_data = data_loader(**data_loader_kwargs)
        assert loaded_data is not None

        if output_type is None or data_loader_kwargs['as_numpy']:
            self.data = loaded_data
        elif output_type == 'numpy':
            self.data = to_numpy(loaded_data, K)
        else:
            raise NotImplementedError(output_type)

        return self.data

    def load_model(self, model, call=False, **model_kwargs):
        """
        Load the model. Takes a model object, or a pipeline that downloads & configures before returning a model object.

        :param model: model object, e.g., a tf.keras.Sequential, tl.Serial,  nn.Module instance

        :param call: call `model()` even if `len(model_kwargs) == 0`
        :type call: ```bool```

        :param \**model_kwargs: to be passed into the model. If empty, doesn't call, unless call=True.
           to be passed into the model. If empty, doesn't call, unless call=True.

        :return self.model, e.g., the result of applying `model_kwargs` on model

        :Keyword Arguments:
            * *num_classes* (``int``) --
              Number of classes
        """

        self.model = model if len(model_kwargs) == 0 and not call else model(**model_kwargs)
        return self.model

    @abstractmethod
    def train(self, callbacks, epochs, loss, metrics, metric_emit_freq, optimizer,
              save_directory, output_type='infer', writer=stdout, *args, **kwargs):
        """
        Run the training loop for your ML pipeline.

        :param callbacks: Collection of callables that are run inside the training loop
        :type callbacks: ```None or List[Callable] or Tuple[Callable]```

        :param epochs: number of epochs (must be greater than 0)
        :type epochs: ```int```

        :param loss: Loss function, can be a string (depending on the framework) or an instance of a class
        :type loss: ```str or Callable or Any```

        :param metrics: Collection of metrics to monitor, e.g., accuracy, f1
        :type metrics: ```None or List[Callable or str] or Tuple[Callable or str]```

        :param metric_emit_freq: Frequency of metric emission, e.g., `lambda: epochs % 10 == 0`, defaults to every epoch
        :type metric_emit_freq: ```None or (*args, **kwargs) -> bool```

        :param optimizer: Optimizer, can be a string (depending on the framework) or an instance of a class
        :type callbacks: ```str or Callable or Any```

        :param save_directory: Directory to save output in, e.g., weights in h5 files. If None, don't save.
        :type save_directory: ```None or str```

        :param output_type: `if save_directory is not None` then save in this format, e.g., 'h5'.
        :type output_type: ```str```

        :param writer: Writer for all output, could be a TensorBoard instance, a file handler like stdout or stderr
        :type writer: ```stdout or Any```

        :param \*args:
        :param \**kwargs:
        :return:
        """
        assert epochs is not None and epochs > 0


del ABC, abstractmethod, stdout, Tuple, Any, List, tf, np

__all__ = ['BaseTrainer']
