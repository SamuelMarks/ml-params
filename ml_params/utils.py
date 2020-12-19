"""
Collection of utility functions
"""
from inspect import getmembers
from os import path
from sys import version_info


def camel_case(st, upper=False):
    """
    Convert string to camel-case (upper or lower)

    :param st: input string
    :type st: ```str```

    :param upper: upper camelcase if True, else lower camelcase
    :type upper: ```bool```

    :return: camel case representation of input string
    :rtype: ```str```
    """
    output = "".join(x for x in st.title() if x.isalnum())
    return getattr(output[0], "upper" if upper else "lower")() + output[1:]


def common_dataset_handler(
    ds_builder, scale, K, as_numpy, **download_and_prepare_kwargs
):
    """
    Helper function that is to be used by the different dataset builders

    :param ds_builder: dataset builder
    :type ds_builder: ```Union[tfds.core.DatasetBuilder, Tuple[tf.data.Dataset, tf.data.Dataset],
     Tuple[np.ndarray, np.ndarray]```

    :param scale: rescale input (divide) by this amount, None for do nothing
    :type scale: ```Optional[Union[int, float]]```

    :param K: backend engine, e.g., `np` or `tf`
    :type K: ```Literal['np', 'tf']```

    :param as_numpy: Convert to numpy ndarrays
    :type as_numpy: ```bool```

    :param download_and_prepare_kwargs:
    :type download_and_prepare_kwargs: ```**download_and_prepare_kwargs```

    :return: Train and tests dataset splits
    :rtype: ```Union[Tuple[tf.data.Dataset,tf.data.Dataset,tfds.core.DatasetInfo], Tuple[np.ndarray,np.ndarray,Any]]```
    """
    as_dataset_kwargs, info = {"batch_size": -1}, None
    if hasattr(ds_builder, "download_and_prepare") and hasattr(
        ds_builder, "as_dataset"
    ):
        train_ds, test_ds, dl_and_prep = None, None, True
        if (
            "download_config" in download_and_prepare_kwargs
            and download_and_prepare_kwargs["download_config"].manual_dir
        ):
            dl_and_prep = not path.isdir(ds_builder._data_dir)
            if dl_and_prep:
                name_slash = "{}{}{}".format(path.sep, ds_builder.name, path.sep)
                other_data_dir = ds_builder._data_dir.replace(
                    name_slash, "{}downloads{}".format(path.sep, name_slash)
                )
                dl_and_prep = not path.isdir(other_data_dir)
                if not dl_and_prep:
                    ds_builder._data_dir = other_data_dir

            if not dl_and_prep:
                import tensorflow_datasets.public_api as tfds

                info = ds_builder.info
                ds_builder = tfds.builder(
                    ds_builder.name,
                    data_dir=path.dirname(path.dirname(ds_builder._data_dir)),
                )
                as_dataset_kwargs.update({"as_supervised": True, "batch_size": 1})
        if dl_and_prep:
            ds_builder.download_and_prepare(**download_and_prepare_kwargs)

        if train_ds is None:
            train_ds = ds_builder.as_dataset(split="train", **as_dataset_kwargs)
        if test_ds is None:
            test_ds = ds_builder.as_dataset(split="test", **as_dataset_kwargs)
    elif hasattr(ds_builder, "train_stream") and hasattr(ds_builder, "eval_stream"):
        return ds_builder  # Handled elsewhere, this is from trax
    else:
        train_ds, test_ds = ds_builder

    if as_numpy:
        train_ds, test_ds = to_numpy(train_ds, K), to_numpy(test_ds, K)

    if K is not None and scale is not None:
        if isinstance(scale, tuple):
            assert scale[0] == scale[1]
            scale = scale[0]
        train_ds["image"] = K.float32(train_ds["image"]) / scale
        test_ds["image"] = K.float32(test_ds["image"]) / scale

    return train_ds, test_ds, info or train_ds._info


def to_numpy(obj, K=None, device=None):
    """
    Convert input to numpy

    :param obj: Any input that can be converted to numpy (raises error otherwise)
    :type obj: ```Any```

    :param K: backend engine, e.g., `np` or `tf`; defaults to `np`
    :type K: ```Literal['np', 'tf']```

    :param device: The (optional) Device to which x should be transferred.
      If given, then the result is committed to the device.
      If the device parameter is None, then this operation behaves like the identity function
      if the operand is on any device already, otherwise it transfers the data to the default device, uncommitted.
    :type device: ```Optional[Device]```

    :return: numpy type, probably np.ndarray
    :rtype: ```np.ndarray```
    """
    module_name = "numpy" if K is None else K.__name__

    if obj is None:
        return None if K is None else K.nan
    elif type(obj).__module__ == module_name:
        return obj
    elif hasattr(obj, "as_numpy"):
        return obj.as_numpy()
    elif hasattr(obj, "numpy"):
        return obj.numpy()
    elif isinstance(obj, dict) and "image" in obj and "label" in obj:
        if module_name == "jax.numpy":

            def __to_numpy(o, _K=None):
                """
                Convert input to a DeviceArray

                :param o: An object with a `numpy` method
                :type o: ```Any```

                :param _K: backend engine, e.g., `np` or `tf`; defaults to `np`
                :type _K: ```Literal['np', 'tf']```

                :return: The array on the device
                :rtype: ```DeviceArray```
                """
                import jax

                return jax.device_put(o.numpy(), device=device)

        else:
            __to_numpy = _to_numpy

        return {
            "image": __to_numpy(obj["image"], K),
            "label": __to_numpy(obj["label"], K),
        }
    elif type(obj).__name__ == "PrefetchDataset":
        # ^`isinstance` said `arg 2 must be a type or tuple of types`
        import tensorflow_datasets as tfds

        return tfds.as_numpy(obj)

    raise TypeError("Unable to convert {!r} to numpy".format(type(obj)))


# Alias need unlike in JavaScript where you have proper hoisting
_to_numpy = to_numpy


def to_d(obj):
    """
    Convert the input to a dictionary

    :param obj: input value. Will have `dir` run against it if not a dict.
    :type obj: ```Union[dict, Any]```

    :return: Dictionary representation of input
    :rtype: ```dict```
    """
    return (
        obj
        if isinstance(obj, dict)
        else dict(
            filter(lambda key_inst: not key_inst[0].startswith("_"), getmembers(obj))
        )
    )


# The next 2 functions are from https://stackoverflow.com/a/1653248
def parse_to_argv_gen(s):
    """
    Generate a sys.argv style parse of the input string

    :param s: Input string
    :type s: ```str```

    :return: Generator of tokens; like in sys.argv
    :rtype: ```Iterator[str]```
    """
    _QUOTE_CHARS_DICT = {
        "\\": "\\",
        " ": " ",
        '"': '"',
        "r": "\r",
        "n": "\n",
        "t": "\t",
    }

    quoted, s_iter, join_string, c_list, c = False, iter(s), s[0:0], [], " "
    err = "Bytes must be decoded to Unicode first"

    while True:
        # Skip whitespace
        try:
            while True:
                assert isinstance(c, str) and version_info[0] >= 3, err
                if not c.isspace():
                    break
                c = next(s_iter)
        except StopIteration:
            break
        # Read word
        try:
            while True:
                assert isinstance(c, str) and version_info[0] >= 3, err
                if not quoted and c.isspace():
                    break
                if c == '"':
                    quoted, c = not quoted, None
                elif c == "\\":
                    c = _QUOTE_CHARS_DICT.get(next(s_iter))
                if c is not None:
                    c_list.append(c)
                c = next(s_iter)
            yield join_string.join(c_list)
            c_list.clear()
        except StopIteration:
            yield join_string.join(c_list)
            break


def parse_to_argv(s):
    """
    Do a sys.argv style parse of the input string

    :param s: Input string
    :type s: ```str```

    :return: List of tokens; like in sys.argv
    :rtype: ```List[str]```
    """
    return list(parse_to_argv_gen(s))


__all__ = ["camel_case", "common_dataset_handler", "parse_to_argv", "to_d", "to_numpy"]
