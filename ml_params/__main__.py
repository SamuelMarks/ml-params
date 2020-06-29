# !/usr/bin/env python

from argparse import ArgumentParser
from enum import Enum
from importlib import import_module

from pkg_resources import working_set

from ml_params import __version__
from ml_params.argparse_extras import EnumAction
from ml_params.base import BaseTrainer

engines = tuple(filter(lambda p: p.replace('-', '_').startswith('ml_params_'),
                       map(lambda p: p.project_name, working_set)))
engine_enum = tuple(map(lambda p: (lambda q: (q.title(), q))(p[p.rfind('-') + 1:]), engines))


def _build_parser():
    parser = ArgumentParser(
        prog='python -m ml_params',
        description='Consistent CLI API for JAX, Trax, TensorFlow, Keras, and PyTorch'
    )
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(__version__))

    parser.add_argument('--engine', type=Enum('EngineEnum', engine_enum),
                        action=EnumAction, required=True)

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', type=int)

    parser.add_argument('--generate',
                        help='Directory where the input files are partitioned. Can be symlinked. '
                             'Example contents: "ds_name/train/a.jpg"; "ds_name/test/b.jpg".',
                        # type=PathType(exists=True, type='dir')
                        )

    parser.add_argument('--tfds', help='Directory for TFrecords and other metadata, '
                                       'whence TensorFlow Datasets are prepared.',
                        # type=PathType(exists=True, type='dir')
                        )
    return parser


if __name__ == '__main__':
    _parser = _build_parser()
    args = _parser.parse_args()

    if args.train:
        trainer = getattr(
            import_module('{engine}.ml_params_impl'.format(
                engine='ml_params_{engine}'.format(engine=args.engine.value)
            )),
            '{upper_engine}Trainer'.format(upper_engine=args.engine.name)
        )()  # type: BaseTrainer
        trainer.train(epochs=args.epochs)
    else:
        _parser.error('--train must be specified. Maybe in future this CLI will do more?')
