# !/usr/bin/env python
"""
CLI interface
"""

import sys
from argparse import ArgumentParser
from collections import OrderedDict, deque
from enum import Enum
from importlib import import_module
from operator import itemgetter
from os import environ

from argparse_utils.actions.enum import EnumAction
from pkg_resources import working_set

from ml_params import __version__
from ml_params.base import BaseTrainer

if sys.version[0] == "3":
    string_types = (str,)
else:
    string_types = (basestring,)

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

    :return: instanceof ArgumentParser
    :rtype: ```ArgumentParser```
    """

    parser = ArgumentParser(
        prog="python -m ml_params",
        description="Consistent CLI for every popular ML framework.",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s {}".format(__version__)
    )

    parser.add_argument(
        "--engine",
        type=Enum("EngineEnum", engine_enum),
        action=EnumAction,
        help='ML engine, e.g., "TensorFlow", "JAX", "pytorch"',
    )

    return parser


def get_one_arg(args, argv=None):
    """
    Hacked together parser to get just one value

    :param args:
    :type args: ```Tuple[str]```

    :param argv: Defaults to `sys.argv`
    :type argv: ```Optional[List[str]]```

    :rtype: ```Optional[str]```
    """
    engine_val, next_is_sym = None, None
    for e in argv or sys.argv:
        for eng in args:
            if e.startswith(eng):
                if e == eng:
                    next_is_sym = eng
                else:
                    return e[len(eng) + 1 :]
            elif next_is_sym == eng:
                return e


if __name__ == "__main__":
    engine_name = engine = get_one_arg(("-e", "--engine")) or environ.get(
        "ML_PARAMS_ENGINE"
    )

    if any(filter(lambda eng: eng == engine, map(itemgetter(1), engine_enum))):
        engine = import_module(
            "{engine_fqdn}.ml_params.cli".format(
                engine_fqdn="ml_params_{engine_name}".format(engine_name=engine_name)
            )
        )

    _parser = _build_parser()

    if isinstance(engine, (type(None), string_types)):
        _parser.print_help()
        _parser.error(
            "--engine must be provided, and from installed ml-params-* options"
        )

    trainer_mod = import_module(
        "{engine_fqdn}.ml_params.trainer".format(
            engine_fqdn="ml_params_{engine_name}".format(engine_name=engine_name)
        )
    )
    Trainer = getattr(
        trainer_mod, next(filter(lambda c: c.endswith("Trainer"), trainer_mod.__all__))
    )

    trainer: BaseTrainer = Trainer()

    subparsers = _parser.add_subparsers(
        help="subcommand to run. Hacked to be chainable.", dest="command"
    )

    deque(
        (
            getattr(engine, func_name)(
                subparsers.add_parser(func_name[: -len("_parser")])
            )
            for func_name in sorted(engine.__all__)
            if func_name.endswith("_parser")
        ),
        maxlen=0,
    )

    rest = sys.argv[1:]
    while len(rest):
        args, rest = _parser.parse_known_args(rest)

        command = args.command
        getattr(trainer, command)(
            **{k: v for k, v in vars(args).items() if v is not None and k != "command"}
        )
