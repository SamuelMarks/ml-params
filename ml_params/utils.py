def camel_case(st, upper=False):
    output = ''.join(x for x in st.title() if x.isalnum())
    return getattr(output[0], 'upper' if upper else 'lower')() + output[1:]


def common_dataset_handler(ds_builder, download_and_prepare_kwargs, scale, K, as_numpy):
    """
    Helper function that is to be used by the different dataset builders

    :param ds_builder:
    :type ds_builder: ```tfds.core.DatasetBuilder or Tuple[tf.data.Dataset, tf.data.Dataset]
                         or Tuple[np.ndarray, np.ndarray]```

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
    if hasattr(ds_builder, 'download_and_prepare') and hasattr(ds_builder, 'as_dataset'):
        ds_builder.download_and_prepare(**(download_and_prepare_kwargs or {}))

        train_ds = ds_builder.as_dataset(split='train', batch_size=-1)
        test_ds = ds_builder.as_dataset(split='test', batch_size=-1)
    elif hasattr(ds_builder, 'train_stream') and hasattr(ds_builder, 'eval_stream'):
        return ds_builder  # Handled elsewhere, this is from trax
    else:
        train_ds, test_ds = ds_builder

    if as_numpy:
        train_ds, test_ds = train_ds.numpy(), test_ds.numpy()

    if K is not None and scale is not None:
        if isinstance(scale, tuple):
            assert scale[0] == scale[1]
            scale = scale[0]
        train_ds['image'] = K.float32(train_ds['image']) / scale
        test_ds['image'] = K.float32(test_ds['image']) / scale

    return train_ds, test_ds
