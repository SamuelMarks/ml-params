from functools import partial

from ml_prepare.datasets import datasets2classes

from ml_params.tf_utils import get_from_tensorflow_datasets
from ml_params.utils import common_dataset_handler


def load_data_from_ml_prepare(dataset_name, tfds_dir=None, generate_dir=None,
                              retrieve_dir=None, K=None, as_numpy=False, scale=None):
    """
    Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

    :param dataset_name: name of dataset
    :type dataset_name: ```str```

    :param tfds_dir: directory to look for models in. Default is ~/tensorflow_datasets.
    :type tfds_dir: ```None or str```

    :param generate_dir:
    :type generate_dir: ```None or str```

    :param retrieve_dir:
    :type retrieve_dir: ```None or str```

    :param K: backend engine, e.g., `np` or `tf`
    :type K: ```None or np or tf or Any```

    :param as_numpy: Convert to numpy ndarrays
    :type as_numpy: ```bool```

    :param scale: scale (height, width)
    :type scale: ```(int, int)```

    :return: Train and tests dataset splits
    :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
    """
    from ml_prepare.exectors import build_tfds_dataset

    assert dataset_name in datasets2classes

    ds_builder = build_tfds_dataset(
        dataset_name=dataset_name,
        tfds_dir=tfds_dir,
        generate_dir=generate_dir,
        retrieve_dir=retrieve_dir,
        **({} if scale is None else {
            'image_height': scale[0],
            'image_width': scale[1]
        })
    )

    if hasattr(ds_builder, 'download_and_prepare_kwargs'):
        download_and_prepare_kwargs = getattr(ds_builder, 'download_and_prepare_kwargs')
        delattr(ds_builder, 'download_and_prepare_kwargs')
    else:
        download_and_prepare_kwargs = {}

    return common_dataset_handler(
        ds_builder=ds_builder,
        scale=None,  # Keep this as None, the processing is done above
        K=K, as_numpy=as_numpy,
        **download_and_prepare_kwargs
    )


def load_data_from_tfds_or_ml_prepare(dataset_name, tfds_dir=None,
                                      K=None, as_numpy=False, **data_loader_kwargs):
    """
    Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

    :param dataset_name: name of dataset
    :type dataset_name: ```str```

    :param tfds_dir: directory to look for models in. Default is ~/tensorflow_datasets.
    :type tfds_dir: ```None or str```

    :param K: backend engine, e.g., `np` or `tf`
    :type K: ```None or np or tf or Any```

    :param as_numpy: Convert to numpy ndarrays
    :type as_numpy: ```bool```

    :param data_loader_kwargs: pass this as arguments to data_loader function
    :type data_loader_kwargs: ```**data_loader_kwargs```

    :return: Train and tests dataset splits
    :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
    """
    from ml_prepare.exectors import build_tfds_dataset

    ds_builder = (partial(build_tfds_dataset, tfds_dir=tfds_dir) if dataset_name in datasets2classes
                  else partial(get_from_tensorflow_datasets, data_dir=tfds_dir))(
        dataset_name=dataset_name,
        **{k: v for k, v in data_loader_kwargs.items()
           if v is not None and k != 'tfds_dir'}
    )

    if hasattr(ds_builder, 'download_and_prepare_kwargs'):
        download_and_prepare_kwargs = getattr(ds_builder, 'download_and_prepare_kwargs')
        delattr(ds_builder, 'download_and_prepare_kwargs')
    else:
        download_and_prepare_kwargs = {}

    return common_dataset_handler(
        ds_builder=ds_builder,
        scale=None, K=K, as_numpy=as_numpy,
        **download_and_prepare_kwargs
    )
