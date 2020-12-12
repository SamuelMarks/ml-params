# !/usr/bin/env python
"""
CLI interface
"""
import sys
from argparse import SUPPRESS, ArgumentParser, HelpFormatter
from collections import deque
from enum import Enum
from functools import partial
from importlib import import_module
from itertools import filterfalse
from operator import itemgetter
from os import environ

from argparse_utils.actions.enum import EnumAction
from pkg_resources import working_set
from yaml import safe_load as loads

from ml_params import __version__
from ml_params.base import BaseTrainer

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


def parse_from_symbol_table(value, dest, symbol_table):
    if dest in symbol_table and isinstance(value, str):
        name, _, raw = value.partition(":")
        config_name = "{name}Config".format(name=name)
        if _ and raw and config_name in symbol_table[dest].__all__:
            return getattr(symbol_table[dest], config_name), raw

    return None, None


class ImportArgumentParser(ArgumentParser):
    """ Attempt at creating an importing argument parser """

    def __init__(
        self,
        symbol_table,
        prog=None,
        usage=None,
        description=None,
        epilog=None,
        parents=None,
        formatter_class=HelpFormatter,
        prefix_chars="-",
        fromfile_prefix_chars=None,
        argument_default=None,
        conflict_handler="error",
        add_help=True,
        allow_abbrev=True,
    ):
        super(ImportArgumentParser, self).__init__(
            prog=prog,
            usage=usage,
            description=description,
            epilog=epilog,
            parents=parents or [],
            formatter_class=formatter_class,
            prefix_chars=prefix_chars,
            fromfile_prefix_chars=fromfile_prefix_chars,
            argument_default=argument_default,
            conflict_handler=conflict_handler,
            add_help=add_help,
            allow_abbrev=allow_abbrev,
        )
        self.symbol_table = symbol_table

    def _check_value(self, action, value):
        """ Check the value, parsing out the config class name if that object is provided """
        super(ImportArgumentParser, self)._check_value(
            action,
            value.__class__.__name__[: -len("Config")]
            if action.choices is not None
            and value not in action.choices
            and not isinstance(value, (str, int, float, complex, list, tuple, set))
            else value,
        )


def _build_parser(symbol_table=None):
    """
    Parser builder

    :return: instanceof ArgumentParser
    :rtype: ```ArgumentParser```
    """
    parser = ImportArgumentParser(
        prog="python -m ml_params",
        description="Consistent CLI for every popular ML framework.",
        symbol_table=symbol_table,
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


def get_one_arg(args, argv):
    """
    Hacked together parser to get just one value

    :param args: Name of arg to retrieve. If multiple specified, return first found.
    :type args: ```Tuple[str]```

    :param argv: Argument list
    :type argv: ```List[str]```

    :return: First matching arg value
    :rtype: ```Optional[str]```
    """
    assert isinstance(args, (tuple, list)), "Expected tuple|list got {!r}".format(
        type(args)
    )
    next_is_sym = None
    for e in argv:
        for eng in args:
            if e.startswith(eng):
                if e == eng:
                    next_is_sym = eng
                else:
                    return e[len(eng) + 1 :]
            elif next_is_sym == eng:
                return e


def main(argv=None):
    """
    Main CLI. Actually perform the argument parsing &etc.

    :param argv: argv, defaults to ```sys.argv```
    :type argv: ```Optional[List[str]]```
    """
    argv = argv or sys.argv[1:]
    engine_name = engine = get_one_arg(("-e", "--engine"), argv) or environ.get(
        "ML_PARAMS_ENGINE"
    )

    if any(filter(lambda eng: eng == engine, map(itemgetter(1), engine_enum))):
        engine = import_module(
            "{engine_fqdn}.ml_params.cli".format(
                engine_fqdn="ml_params_{engine_name}".format(engine_name=engine_name)
            )
        )
        symbol_table = getattr(
            import_module(
                "{engine_fqdn}.ml_params.extra_symbols".format(
                    engine_fqdn="ml_params_{engine_name}".format(
                        engine_name=engine_name
                    )
                )
            ),
            "extra_symbols",
        )
    else:
        symbol_table = None

    _parser = _build_parser(symbol_table=symbol_table)

    if "--version" in argv:
        _parser.parse_args(["--version"])
    elif isinstance(engine, (type(None), string_types)):
        _parser.print_help(file=sys.stderr)
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

    trainer = Trainer()  # type: BaseTrainer

    # Add CLI parsers from dynamically imported library
    subparsers = _parser.add_subparsers(
        help="subcommand to run. Hacked to be chainable.", dest="command"
    )

    def dbg(arg):
        dbg.c += 1
        if dbg.c < 10:
            for attr in dir(arg):
                print(attr, getattr(arg, attr))
        return arg

    dbg.c = 0

    deque(
        (
            print("Adding CLI parser: {!s} ;".format(func_name))
            or getattr(engine, func_name)(
                subparsers.add_parser(
                    func_name[: -len("_parser")], symbol_table=symbol_table
                )
            )
            for func_name in sorted(engine.__all__)
            if func_name.endswith("_parser")
        ),
        maxlen=0,
    )

    # Make required CLI arguments optional iff they are required but have a default value.

    def remove_required(sub_parser_action_idx, argument_parser_name, action_idx):
        """
        Set the required parameter to False

        :param sub_parser_action_idx: index of sub_parser_action
        :type sub_parser_action_idx: ```str```

        :param argument_parser_name: name of argument_parser
        :type argument_parser_name: ```str```

        :param action_idx: index of action
        :type action_idx: ```int```
        """
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

    _parse_from_symbol_table = partial(
        parse_from_symbol_table, symbol_table=symbol_table
    )

    def change_type(sub_parser_action_idx, argument_parser_name, action_name):
        """
        Set the type to construct something from the symbol table

        :param sub_parser_action_idx: index of sub_parser_action
        :type sub_parser_action_idx: ```str```

        :param argument_parser_name: name of argument_parser
        :type argument_parser_name: ```str```

        :param action_name: Name of action
        :type action_name: ```str```
        """
        _parser._subparsers._group_actions[sub_parser_action_idx].choices[
            argument_parser_name
        ]._option_string_actions[action_name].type = partial(
            parse_type,
            dest=_parser._subparsers._group_actions[sub_parser_action_idx]
            .choices[argument_parser_name]
            ._option_string_actions[action_name]
            .dest,
        )

    def parse_type(v, dest):
        init, arguments = _parse_from_symbol_table(v, dest)
        if init is not None and arguments is not None:
            __args = loads(arguments)
            return init(**__args)
        return v

    # When the symbol_table contains the target (dest) CLI parameter,
    # change its type so that it'll parse out and construct the right [configuration] object
    deque(
        map(
            lambda idx_sub_parser_action: deque(
                map(
                    lambda name_argument_parser: deque(
                        map(
                            lambda idx_name: change_type(
                                sub_parser_action_idx=idx_sub_parser_action[0],
                                argument_parser_name=name_argument_parser[0],
                                action_name=idx_name[1],
                            ),
                            filter(
                                lambda idx_name: name_argument_parser[1]
                                ._option_string_actions[idx_name[1]]
                                .dest
                                in symbol_table
                                and name_argument_parser[1]
                                ._option_string_actions[idx_name[1]]
                                .type
                                in frozenset((None, str)),
                                enumerate(
                                    name_argument_parser[1]._option_string_actions
                                ),
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
    rest = argv
    _parser.symbol_table = symbol_table
    while len(rest) != 0:
        args, rest = _parser.parse_known_args(rest)
        # print("__main__::args:", args, ";\n__main__::rest:", rest, ";")

        # if args.command == 'train':
        #    exit(5)

        getattr(trainer, args.command)(
            **{
                k: v
                for k, v in vars(args).items()
                if isinstance(v, dict)
                or v is not None
                and v != "None"
                and k != "command"
            }
        )


def run_main():
    """" Run the `main` function if `__name__ == "__main__"` """
    if __name__ == "__main__":
        main()


run_main()
