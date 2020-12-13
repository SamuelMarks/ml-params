# !/usr/bin/env python
"""
CLI interface
"""
import sys
from argparse import SUPPRESS, ArgumentParser, HelpFormatter
from collections import deque
from functools import partial
from importlib import import_module
from itertools import filterfalse
from operator import attrgetter, eq, itemgetter
from os import environ

from pkg_resources import working_set

from ml_params import __version__
from ml_params.base import BaseTrainer
from ml_params.utils import parse_to_argv

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
        map(attrgetter("project_name"), working_set),
    )
)
engine_enum = tuple(
    map(lambda p: (lambda q: (q.title(), q))(p[p.rfind("-") + 1 :]), engines)
)

name_tpl = "{name}Config"


def parse_from_symbol_table(value, dest, symbol_table):
    """
    Try to acquire the constructor and [unparsed] arguments

    :param value: The value to be checked
    :type value: ```Any```

    :param dest: The name of the attribute to be added to the object returned by `parse_args()`.
    :type dest: ```str```

    :param symbol_table: A mapping from string to an in memory construct, e.g., a class or function.
    :type symbol_table: ```Dict[Str, Any]```

    :return: (name, Constructor, unparsed arguments) if found else (None, None, None)
    :rtype: ```Tuple[Optional[str], Optional[Any], Optional[str]]
    """
    if dest in symbol_table and isinstance(value, str):
        name, _, raw = value.partition(":")
        config_name = name_tpl.format(name=name)
        if _ and raw and config_name in symbol_table[dest].__all__:
            return name, getattr(symbol_table[dest], config_name), raw

    return None, None, None


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
        exit_on_error=True,
    ):
        """
        Construct the argument parser

        :param symbol_table: A mapping from string to an in memory construct, e.g., a class or function.
        :type symbol_table: ```Dict[Str, Any]```

        :param prog: The name of the program
        :type prog: ```Optional[str]```

        :param usage: The string describing the program usage (default: generated from arguments added to parser)
        :type usage: Optional[str]

        :param description: Text to display before the argument help
        :type description: ```Optional[str]```

        :param epilog: Text to display after the argument help
        :type epilog: ```Optional[str]```

        :param parents: A list of `ArgumentParser` objects whose arguments should also be included
        :type parents: ```Sequence[ArgumentParser]```

        :param formatter_class: A class for customizing the help output
        :type formatter_class: ```_FormatterClass```

        :param prefix_chars: The set of characters that prefix optional arguments
        :type prefix_chars: ```str```

        :param fromfile_prefix_chars: The set of characters that prefix files from which
            additional arguments should be read
        :type fromfile_prefix_chars: ```Optional[str]```

        :param argument_default: The global default value for arguments
        :type argument_default: ```Optional[str]```

        :param conflict_handler: The strategy for resolving conflicting optionals (usually unnecessary)
        :type conflict_handler: ```str```

        :param add_help: Add a `-h`/`--help` option to the parser
        :type add_help: ```bool```

        :param allow_abbrev: Allows long options to be abbreviated if the abbreviation is unambiguous
        :type allow_abbrev: ```bool```

        :param exit_on_error: Determines whether or not `ArgumentParser` exits with error info when an error occurs.
            (3.9+ only)
        :type exit_on_error: ```bool```
        """
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
            **dict(exit_on_error=exit_on_error) if sys.version_info[:2] > (3, 8) else {}
        )
        self.symbol_table = symbol_table

    def _check_value(self, action, value):
        """
        Check the value, parsing out the config class name if that object is provided

        :param action: The action for the value being checked
        :type action: ```argparse.Action```

        :param value: The value to be checked
        :type value: ```Any```
        """
        super(ImportArgumentParser, self)._check_value(
            action,
            value.__class__.__name__
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
        type=str,
        choices=tuple(map(itemgetter(1), engine_enum)),
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

    if any(filter(partial(eq, engine), map(itemgetter(1), engine_enum))):
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

    def remove_required(the_parser):
        def _remove_required(sub_parser_action_idx, argument_parser_name, action_idx):
            """
            Set the required parameter to False

            :param sub_parser_action_idx: index of sub_parser_action
            :type sub_parser_action_idx: ```str```

            :param argument_parser_name: name of argument_parser
            :type argument_parser_name: ```str```

            :param action_idx: index of action
            :type action_idx: ```int```
            """
            the_parser._subparsers._group_actions[sub_parser_action_idx].choices[
                argument_parser_name
            ]._actions[action_idx].required = False

        deque(
            map(
                lambda idx_sub_parser_action: deque(
                    map(
                        lambda name_argument_parser: deque(
                            map(
                                lambda idx_action: _remove_required(
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
                enumerate(the_parser._subparsers._group_actions),
            ),
            maxlen=0,
        )

    remove_required(_parser)

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

    def parse_type(type_name, dest):
        """
        Parse the type out of the symbol table (if in, else identity)

        :param type_name: The value of the `type=` from argparse
        :type type_name: ```str```

        :param dest: The name of the attribute to be added to the object returned by `parse_args()`.
        :type dest: ```str```

        :return: Identity or the constructed symbol out of the symbol table (with args)
        :rtype: ```Union[str, Any]```
        """
        name, init, arguments = _parse_from_symbol_table(type_name, dest)
        if init is not None and arguments is not None:
            # if CLI_SUB_SUB_PARSE_TYPE == YAML:
            #     __args = yaml.safe_load(arguments)
            #     return init(**__args)
            # else:
            __parser = ArgumentParser(
                prog="--{dest}".format(dest=dest),
                description="Generated parser for a single parameter",
            )
            __sub = __parser.add_subparsers(
                help="Subcommand for internal compatibility with helper functions"
            )
            ___actual_parser = __sub.add_parser(
                name,
                prog="{parent_prog} '{name}:".format(
                    parent_prog=__parser.prog, name=name
                ),
                help="The actual parser",
            )
            init(___actual_parser)
            remove_required(__parser)
            sub_namespace = __parser.parse_args([name] + parse_to_argv(arguments))
            del __parser, __sub, ___actual_parser
            sub_namespace.__class__.__name__ = name
            return sub_namespace
        return type_name

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
