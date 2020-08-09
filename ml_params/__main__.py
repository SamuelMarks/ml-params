# !/usr/bin/env python
"""
CLI interface
"""

from argparse import ArgumentParser
from enum import Enum
from importlib import import_module

from argparse_utils.actions.enum import EnumAction
from pkg_resources import working_set

from ml_params import __version__
from ml_params.base import BaseTrainer

engines = tuple(
    filter(
        lambda p: p.replace("-", "_").startswith("ml_params_"),
        map(lambda p: p.project_name, working_set),
    )
)
engine_enum = tuple(
    map(lambda p: (lambda q: (q.title(), q))(p[p.rfind("-") + 1 :]), engines)
)


def _build_parser():
    """
    Parser builder

    :returns: instanceof ArgumentParser
    :rtype: ```ArgumentParser```
    """

    parser = ArgumentParser(
        prog="python -m ml_params",
        description="Consistent CLI API for JAX, Trax, TensorFlow, Keras, and PyTorch",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s {}".format(__version__)
    )

    parser.add_argument(
        "--engine",
        type=Enum("EngineEnum", engine_enum),
        action=EnumAction,
        required=True,
    )

    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--callback",
        type=str,
        action="append",
        dest="callbacks",
        required=True,
        help="Collection of callables that are run inside the training loop",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="number of epochs (must be greater than 0)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        required=True,
        help="number of epochs (must be greater than 0)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        action="append",
        dest="metrics",
        required=True,
        help="Collection of metrics to monitor, e.g., accuracy, f1",
    )
    parser.add_argument(
        "--metric_emit_freq",
        type=str,
        required=True,
        help="Frequency of metric emission, e.g., "
        "`lambda epochs: epochs % 10 == 0`, defaults to every epoch",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        action="append",
        dest="optimizers",
        required=True,
        help="Optimizer, can be a string (depending on the framework) or an instance of a class",
    )
    parser.add_argument(
        "--save_directory",
        type=str,
        required=True,
        help="Optimizer, can be a string (depending on the framework) or an instance of a class",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        required=False,
        help="`if save_directory is not None` then save in this format, e.g., 'h5'",
    )
    parser.add_argument(
        "--writer",
        type=Enum(
            "WriterEnum",
            (
                ("STDOUT", "stdout"),
                ("STDERR", "stderr"),
                ("SummaryWriter", "tensorboard.SummaryWriter"),
            ),
        ),
        required=False,
        default="stdout",
        action=EnumAction,
        help="Writer for all output, e.g., STDOUT, STDERR, tensorboard.SummaryWriter",
    )

    parser.add_argument(
        "--generate",
        type=str,
        help="Directory where the input files are partitioned. Can be symlinked. "
        'Example contents: "ds_name/train/a.jpg"; "ds_name/test/b.jpg".',
        # type=PathType(exists=True, type='dir')
    )

    parser.add_argument(
        "--tfds",
        type=str,
        help="Directory for TFrecords and other metadata, "
        "whence TensorFlow Datasets are prepared.",
        # type=PathType(exists=True, type='dir')
    )
    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    args = _parser.parse_args()

    if args.train:
        trainer = getattr(
            import_module(
                "{engine}.ml_params_impl".format(
                    engine="ml_params_{engine}".format(engine=args.engine.value)
                )
            ),
            "{upper_engine}Trainer".format(upper_engine=args.engine.name),
        )()  # type: BaseTrainer
        trainer.train(
            callbacks=args.callbacks,
            epochs=args.epochs,
            loss=args.loss,
            metrics=args.metrics,
            metric_emit_freq=args.metric_emit_freq,
            optimizer=args.optimizer,
            save_directory=args.save_directory,
            output_type=args.output_type,
        )
    else:
        _parser.error(
            "--train must be specified. Maybe in future this CLI will do more?"
        )
