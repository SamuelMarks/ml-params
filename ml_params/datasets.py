from ml_prepare.datasets import datasets2classes

from ml_params.tf_utils import get_from_tensorflow_datasets
from ml_params.utils import common_dataset_handler


def load_data_from_ml_prepare(dataset_name, tensorflow_datasets_dir=None, data_loader_kwargs=None):
    """
    Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

    :param dataset_name: name of dataset
    :type dataset_name: ```str```

    :param tensorflow_datasets_dir: directory to look for models in. Default is ~/tensorflow_datasets.
    :type tensorflow_datasets_dir: ```None or str```

    :param data_loader_kwargs: pass this as arguments to data_loader function
    :type data_loader_kwargs: ```None or dict```

    :return: Train and tests dataset splits
    :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
    """
    from ml_prepare.exectors import build_tfds_dataset

    assert dataset_name in datasets2classes
    if data_loader_kwargs is None:
        data_loader_kwargs = {}
    data_loader_kwargs.update({
        'dataset_name': dataset_name,
        'tfds_dir': tensorflow_datasets_dir
    })
    ds_builder = build_tfds_dataset(**data_loader_kwargs)

    if hasattr(ds_builder, 'download_and_prepare_kwargs'):
        download_and_prepare_kwargs = getattr(ds_builder, 'download_and_prepare_kwargs')
        delattr(ds_builder, 'download_and_prepare_kwargs')
    else:
        download_and_prepare_kwargs = None

    return common_dataset_handler(
        ds_builder=ds_builder,
        download_and_prepare_kwargs=download_and_prepare_kwargs,
        scale=None, K=None, as_numpy=False
    )


def load_data_from_tfds_or_ml_prepare(dataset_name, tensorflow_datasets_dir=None, data_loader_kwargs=None):
    """
    Acquire from the official tensorflow_datasets model zoo, or the ophthalmology focussed ml-prepare library

    :param dataset_name: name of dataset
    :type dataset_name: ```str```

    :param tensorflow_datasets_dir: directory to look for models in. Default is ~/tensorflow_datasets.
    :type tensorflow_datasets_dir: ```None or str```

    :param data_loader_kwargs: pass this as arguments to data_loader function
    :type data_loader_kwargs: ```None or dict```

    :return: Train and tests dataset splits
    :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
    """
    from ml_prepare.exectors import build_tfds_dataset

    if data_loader_kwargs is None:
        data_loader_kwargs = {}
    data_loader_kwargs.update({
        'dataset_name': dataset_name,
        'tfds_dir': tensorflow_datasets_dir
    })
    ds_builder = build_tfds_dataset(**data_loader_kwargs) if dataset_name in datasets2classes \
        else get_from_tensorflow_datasets(dataset_name, data_dir=tensorflow_datasets_dir,
                                          **{k: v for k, v in data_loader_kwargs.items()
                                             if v is not None and k in ('K', 'as_numpy', 'scale')})

    if hasattr(ds_builder, 'download_and_prepare_kwargs'):
        download_and_prepare_kwargs = getattr(ds_builder, 'download_and_prepare_kwargs')
        delattr(ds_builder, 'download_and_prepare_kwargs')
    else:
        download_and_prepare_kwargs = None

    return common_dataset_handler(
        ds_builder=ds_builder,
        download_and_prepare_kwargs=download_and_prepare_kwargs,
        scale=None, K=None, as_numpy=False
    )
