__author__ = 'Ryan J McLaughlin'

"""
SABer command line.
"""
import sys
import argparse
import logging

from saber.commands import (info, recruit)

usage = """
saber <command> [<args>]
** Commands include:
recruit        Recruit environmental reads to reference SAG(s).
** Other commands:
info           Display SABer version and other information.
help           Return this message.
Use '-h' to get subcommand-specific help, e.g.
"""


def main():
    commands = {"recruit": recruit,
                "info": info}
    parser = argparse.ArgumentParser(description='Recruit environmental reads to reference SAG(s).',
                                     add_help=False
                                     )
    parser.add_argument('command', nargs='?')
    input_cmd = sys.argv[1:2]
    if input_cmd == ['-h']:
        input_cmd = ['help']
    args = parser.parse_args(input_cmd)

    if (not args.command) | (args.command == 'help'):
        sys.stderr.write(usage)
        sys.exit(1)

    elif args.command not in commands:
        logging.error('Unrecognized command')
        sys.stderr.write(usage)
        sys.exit(1)

    cmd = commands.get(args.command)
    cmd(sys.argv[2:])
    logging.info("SABer has finished successfully.\n")


if __name__ == '__main__':
    main()
