""" In its own module to simplify usage,
    e.g., in environment where tensorflow and/or tensorflow_datasets aren't installed """

from ml_params.utils import common_dataset_handler


def get_from_tensorflow_datasets(
    dataset_name,
    data_dir=None,
    K=None,
    as_numpy=False,
    scale=255.0,
    **download_and_prepare_kwargs
):
    """
    Acquire from the official tensorflow_datasets model zoo

    :param dataset_name: name of dataset
    :type dataset_name: ```str```

    :param data_dir: Where to look for the datasets, defaults to ~/tensorflow_datasets
    :type data_dir: ```Optional[str]```

    :param K: backend engine, e.g., `np` or `tf`
    :type K: ```Literal['np', 'tf']```

    :param as_numpy: Convert to numpy ndarrays
    :type as_numpy: ```bool```

    :param scale: rescale input (divide) by this amount, None for do nothing
    :type scale: ```Optional[Union[int, float]]```

    :param download_and_prepare_kwargs:
    :type download_and_prepare_kwargs: ```**download_and_prepare_kwargs```

    :return: Train and tests dataset splits
    :rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
    """
    import tensorflow_datasets as tfds

    return common_dataset_handler(
        ds_builder=tfds.builder(dataset_name, data_dir=data_dir),
        scale=scale,
        K=K,
        as_numpy=as_numpy,
        **download_and_prepare_kwargs
    )
