# !/usr/bin/env python
"""
CLI interface
"""

import sys
from argparse import ArgumentParser, SUPPRESS
from collections import deque
from enum import Enum
from importlib import import_module
from itertools import filterfalse
from operator import itemgetter
from os import environ

from argparse_utils.actions.enum import EnumAction
from ml_params import __version__
from ml_params.base import BaseTrainer
from pkg_resources import working_set

if sys.version[0] == "3":
    string_types = (str,)
else:
    try:
        string_types = (basestring,)
    except NameError:
        string_types = (str,)

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
    next_is_sym = None
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

    if "--version" in sys.argv[1:]:
        _parser.parse_args(["--version"])
    elif isinstance(engine, (type(None), string_types)):
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

    # Add CLI parsers from dynamically imported library
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

    # Make required CLI arguments optional iff they are required but have a default value.

    def remove_required(sub_parser_action_idx, argument_parser_name, action_idx):
        _parser._subparsers._group_actions[sub_parser_action_idx].choices[
            argument_parser_name
        ]._actions[action_idx].required = False

    deque(
        map(
            lambda idx_sub_parser_action: deque(
                map(
                    lambda name_argument_parser: deque(
                        map(
                            lambda idx_action: remove_required(
                                sub_parser_action_idx=idx_sub_parser_action[0],
                                argument_parser_name=name_argument_parser[0],
                                action_idx=idx_action[0],
                            ),
                            filterfalse(
                                lambda idx_action: idx_action[1].default
                                in frozenset((None, SUPPRESS)),
                                enumerate(name_argument_parser[1]._actions),
                            ),
                        ),
                        maxlen=0,
                    ),
                    idx_sub_parser_action[1].choices.items(),
                ),
                maxlen=0,
            ),
            enumerate(_parser._subparsers._group_actions),
        ),
        maxlen=0,
    )

    # Parse the CLI input continuously—i.e., for each subcommand—until completion. `trainer` holds/updates state.
    rest = sys.argv[1:]
    while len(rest) != 0:
        args, rest = _parser.parse_known_args(rest)

        getattr(trainer, args.command)(
            **{
                k: v
                for k, v in vars(args).items()
                if isinstance(v, dict)
                or v not in frozenset((None, "None"))
                and k != "command"
            }
        )
