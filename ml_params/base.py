"""
Base that is implemented by each child repo, e.g., ml-params-tensorflow, ml-params-pytorch
"""
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from sys import stdout
from typing import Any, List, Optional, Tuple, Union

from ml_params.datasets import load_data_from_tfds_or_ml_prepare___ml_params
from ml_params.utils import to_d, to_numpy

try:
    import numpy as np
except ImportError:
    np = None  # `np` is only used for typings in this file

try:
    import tensorflow as tf
except ImportError:
    tf = None  # `tf` is only used for typings in this file


class BaseTrainer(ABC):
    """
    Trainer that is to be implemented for each ML framework
    """

    data = (
        None
    )  # type: Optional[Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]]
    model = (
        None
    )  # type: Optional[Union[Any, tf.keras.models.Sequential, tf.keras.models.Model]]

    def load_data_c(self, config):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param config: object constructed with all the relevant arguments for `load_data`
        :type config: ```Union[dict, Config, Any]```

        :return: a call to .load_data with the config as params
        :rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
        """
        return self.load_data(**to_d(config))

    def load_data(
        self,
        *,
        dataset_name: str,
        data_loader=load_data_from_tfds_or_ml_prepare___ml_params,
        data_type="infer",
        output_type=None,
        K=None,
        **data_loader_kwargs
    ):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param *: syntactic note indicating everything after is a keyword-only argument

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_loader: function that returns the expected data type.
         Defaults to TensorFlow Datasets and ml_prepare combined one.
        :type data_loader: ```Callable[[...], Union[Tuple[tf.data.Dataset, tf.data.Dataset],
         Tuple[np.ndarray, np.ndarray], Tuple[Any, Any]]```

        :param data_type: incoming data type
        :type data_type: ```str```

        :param output_type: outgoing data_type
        :type output_type: ```Optional[Literal['numpy']]```

        :param K: backend engine, e.g., `np` or `tf`
        :type K: ```Literal['np', 'tf']```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```**data_loader_kwargs```

        :return: Dataset splits (by default, your train and test)
        :rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
        """
        assert dataset_name is not None
        if data_type != "infer":
            raise NotImplementedError(data_type)
        elif data_loader is None:
            data_loader = load_data_from_tfds_or_ml_prepare___ml_params

        data_loader_kwargs.update(
            {
                "dataset_name": dataset_name,
                "K": K,
                "as_numpy": data_loader_kwargs.get("as_numpy", True),
            }
        )

        loaded_data = data_loader(**data_loader_kwargs)
        assert loaded_data is not None

        if output_type is None or data_loader_kwargs["as_numpy"]:
            self.data = loaded_data
        elif output_type == "numpy":
            self.data = tuple(
                chain.from_iterable(
                    (map(partial(to_numpy, K=K), loaded_data[:2]), (loaded_data[2],))
                )
            )
        else:
            raise NotImplementedError(output_type)

        return self.data

    def load_model_c(self, config):
        """
        Load the model.
        Takes a model object, or a pipeline that downloads & configures before returning a model object.

        :param config: object constructed with all the relevant arguments for `load_model`
        :type config: ```Union[dict, Config, Any]```

        :return: a call to .load_model with the config as params
        :rtype: ```load_model```
        """
        return self.load_model(**to_d(config))

    def load_model(self, *, model, call=False, **model_kwargs):
        """
        Load the model.
        Takes a model object, or a pipeline that downloads & configures before returning a model object.

        :param *: syntactic note indicating everything after is a keyword-only argument

        :param model: model object, e.g., a tf.keras.Sequential, tl.Serial,  nn.Module instance
        :type model: ```Any```

        :param call: whether to call `model()` even if `len(model_kwargs) == 0`
        :type call: ```bool```

        :param **model_kwargs: to be passed into the model. If empty, doesn't call, unless call=True.
           to be passed into the model. If empty, doesn't call, unless call=True.
        :type **model_kwargs: ```**model_kwargs```

        :return: Function returning the model, e.g., the result of applying `model_kwargs` on model
        :rtype: ```Callable[[], Any]```
        """

        def get_model():
            """
            Call this to get the model.
            Distributed strategies need models to be constructed within its scope,
            so that's why this function

            :return: model, e.g., the result of applying `model_kwargs` on model
            :rtype: ```Any```
            """
            if not callable(model) or isinstance(model, str):
                get_model.call = False
            self.model = (
                model
                if len(model_kwargs) == 0 or get_model.call is False
                else model(**model_kwargs)
            )
            return self.model

        get_model.call = call

        self.get_model = get_model
        return self.get_model

    def train_c(self, config):
        """
        Run the training loop for your ML pipeline.

        :param config: object constructed with all the relevant arguments for `train`
        :type config: ```Union[dict, Config, Any]```

        :return: a call to .train with the config as params
        :rtype: ```train```
        """
        return self.train(**to_d(config))

    @abstractmethod
    def train(self, *, epochs):
        """
        Run the training loop for your ML pipeline.

        :param *: syntactic note indicating everything after is a keyword-only argument

        :param epochs: number of epochs (must be greater than 0)
        :type epochs: ```int```
        """
        # :raises AssertionError: Whence `epochs is None or < 1`
        assert isinstance(epochs, int) and epochs > 0


del ABC, abstractmethod, stdout, List

__all__ = ["BaseTrainer"]
