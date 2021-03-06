"""
Collection of utility functions
"""
from copy import deepcopy
from functools import partial
from inspect import getmembers
from operator import itemgetter
from os import environ, path
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
    ds_builder,
    scale,
    K,
    as_numpy,
    acquire_and_concat_validation_to_train=True,
    **download_and_prepare_kwargs
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

    :param acquire_and_concat_validation_to_train: Whether to acquire the validation split
      and then concatenate it to train

    :param download_and_prepare_kwargs:
    :type download_and_prepare_kwargs: ```**download_and_prepare_kwargs```

    :return: Train and tests dataset splits
    :rtype: ```Union[Tuple[tf.data.Dataset,tf.data.Dataset,tfds.core.DatasetInfo], Tuple[np.ndarray,np.ndarray,Any]]```
    """
    as_dataset_kwargs, info = {"batch_size": -1}, None
    if hasattr(ds_builder, "download_and_prepare") and hasattr(
        ds_builder, "as_dataset"
    ):
        info, test_ds, train_ds = _handle_tfds(
            acquire_and_concat_validation_to_train,
            as_dataset_kwargs,
            download_and_prepare_kwargs,
            ds_builder,
            info,
        )
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


def _handle_tfds(
    acquire_and_concat_validation_to_train,
    as_dataset_kwargs,
    download_and_prepare_kwargs,
    ds_builder,
    info,
):
    """
    Helper function that is to be used by the different dataset builders

    :param acquire_and_concat_validation_to_train: Whether to acquire the validation split
      and then concatenate it to train
    :type acquire_and_concat_validation_to_train: ```bool```

    :param as_dataset_kwargs:
    :type as_dataset_kwargs: ```**as_dataset_kwargs```

    :param download_and_prepare_kwargs:
    :type download_and_prepare_kwargs: ```**download_and_prepare_kwargs```

    :param ds_builder: dataset builder
    :type ds_builder: ```tfds.core.DatasetBuilder```

    :param info: Dataset info
    :type info: ```tfds.core.DatasetInfo```

    :return: Train and tests dataset splits
    :rtype: ```Union[Tuple[tf.data.Dataset,tf.data.Dataset,tfds.core.DatasetInfo], Tuple[np.ndarray,np.ndarray,Any]]```
    """
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
                data_dir=environ.get(
                    "TFDS_DATA_DIR",
                    path.dirname(path.dirname(ds_builder._data_dir)),
                ),
            )
            as_dataset_kwargs.update({"as_supervised": True, "batch_size": 1})
    if dl_and_prep:
        ds_builder.download_and_prepare(**download_and_prepare_kwargs)
    if train_ds is None:
        train_ds = ds_builder.as_dataset(split="train", **as_dataset_kwargs)
        valid_ds_key = next(
            filter(partial(str.startswith, "valid"), ds_builder.info.splits), None
        )
        if valid_ds_key and acquire_and_concat_validation_to_train:
            print("train was", train_ds.cardinality())
            valid_ds = ds_builder.as_dataset(split=valid_ds_key, **as_dataset_kwargs)
            print("validation is", valid_ds.cardinality())
            train_ds = train_ds.concatenate(valid_ds)
            print("train now", train_ds.cardinality())
    if test_ds is None:
        test_ds = ds_builder.as_dataset(split="test", **as_dataset_kwargs)
    return info, test_ds, train_ds


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


def pop_at_index(
    input_list, key, default=None, process_key=lambda k: k, process_val=lambda v: v
):
    """
    If key in index, remove it from list, and return it

    :param input_list: Input list
    :type input_list: ```list```

    :param key: Lookup key
    :type key: ```str```

    :param default: The default value if key not in l
    :type default: ```Optional[Any]```

    :param process_key: Postprocess the key
    :type process_key: ```Callable[[Any], Any]```

    :param process_val: Postprocess the val
    :type process_val: ```Callable[[Any], Any]```

    :return: default if not in list, else the value from the list (and list is now minus that elem)
    :rtype: ```Optional[Any]```
    """
    # if process_key is not None and not isinstance(key, tuple):
    #    return default
    try:
        if process_key:
            idx = next(
                map(
                    itemgetter(0),
                    filter(
                        None,
                        filter(
                            lambda idx_e: process_key(idx_e[1]) == key,
                            enumerate(input_list),
                        ),
                    ),
                )
            )
        else:
            idx = input_list.index(key)
    except (ValueError, StopIteration):
        if isinstance(default, (list, tuple)) and len(default) == 1:
            return default[0]
        return default
    else:
        return deepcopy(process_val(input_list.pop(idx)))


def set_attr(object, attribute, value):
    """
    Sets the named attribute on the given object to the specified value. Then returns it.

    setattr(x, 'y', v) is equivalent to ``x.y = v''

    :param object: The object
    :type object: ```Any```

    :param attribute: The attribute
    :type attribute: ```str```

    :param value: The value
    :type value: ```Any```
    """
    setattr(object, attribute, value)
    return object


__all__ = [
    "camel_case",
    "common_dataset_handler",
    "parse_to_argv",
    "pop_at_index",
    "set_attr",
    "to_d",
    "to_numpy",
]
