from importlib import import_module


def camel_case(st, upper=False):
    """
    Convert string to camel-case (upper or lower)

    :param st: input string
    :type st: ```str```

    :param upper: upper camelcase if True, else lower camelcase
    :type upper: ```bool```

    :return: st
    :rtype: ```str```
    """
    output = ''.join(x for x in st.title() if x.isalnum())
    return getattr(output[0], 'upper' if upper else 'lower')() + output[1:]


def common_dataset_handler(ds_builder, scale, K, as_numpy, **download_and_prepare_kwargs):
    """
    Helper function that is to be used by the different dataset builders

    :param ds_builder:
    :type ds_builder: ```tfds.core.DatasetBuilder or Tuple[tf.data.Dataset, tf.data.Dataset]
                         or Tuple[np.ndarray, np.ndarray]```

    :param scale: rescale input (divide) by this amount, None for do nothing
    :type scale: ```int or float or None```

    :param K: backend engine, e.g., `np` or `tf`
    :type K: ```None or np or tf or Any```

    :param as_numpy: Convert to numpy ndarrays
    :type as_numpy: ```bool```

    :param download_and_prepare_kwargs:
    :type download_and_prepare_kwargs: ```**download_and_prepare_kwargs```

    :return: Train and tests dataset splits
    :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
    """
    if hasattr(ds_builder, 'download_and_prepare') and hasattr(ds_builder, 'as_dataset'):
        ds_builder.download_and_prepare(**download_and_prepare_kwargs)

        train_ds = ds_builder.as_dataset(split='train', batch_size=-1)
        test_ds = ds_builder.as_dataset(split='test', batch_size=-1)
    elif hasattr(ds_builder, 'train_stream') and hasattr(ds_builder, 'eval_stream'):
        return ds_builder  # Handled elsewhere, this is from trax
    else:
        train_ds, test_ds = ds_builder

    if as_numpy:
        train_ds, test_ds = to_numpy(train_ds, K), to_numpy(test_ds, K)

    if K is not None and scale is not None:
        if isinstance(scale, tuple):
            assert scale[0] == scale[1]
            scale = scale[0]
        train_ds['image'] = K.float32(train_ds['image']) / scale
        test_ds['image'] = K.float32(test_ds['image']) / scale

    return train_ds, test_ds


def to_numpy(obj, K=None, device=None):
    """
    Convert input to numpy

    :param obj: Any input that can be converted to numpy (raises error otherwise)
    :type obj: ```Any```

    :param K: backend engine, e.g., `np` or `tf`; defaults to `np`
    :type K: ```None or np or tf or Any```

    :param device: The (optional) Device to which x should be transferred. If given, then the result is committed to the device. If the device parameter is None, then this operation behaves like the identity function if the operand is on any device already, otherwise it transfers the data to the default device, uncommitted.
    :type device: ```None or Device```

    :return: numpy type, probably np.ndarray
    :rtype: ```np.ndarray```
    """
    module_name = 'numpy' if K is None else K.__name__

    if obj is None:
        return K.nan
    elif type(obj).__module__ == module_name:
        return obj
    elif hasattr(obj, 'as_numpy'):
        return obj.as_numpy()
    elif hasattr(obj, 'numpy'):
        return obj.numpy()
    elif isinstance(obj, dict) and 'image' in obj and 'label' in obj:
        if module_name == 'jax.numpy':
            def to_numpy(o, engine=None, dev=None):
                return import_module('jax').device_put(o.numpy(), device=dev)

        return {'image': to_numpy(obj['image'], K), 'label': to_numpy(obj['label'], K)}

    raise TypeError('Unable to convert {!r} to numpy'.format(type(obj)))
