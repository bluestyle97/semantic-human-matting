import argparse
import warnings
warnings.filterwarnings('ignore')
from utils.config import *

from agents import SHMAgent


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--config',
        default=None,
        help='The Path of configuration file in yaml format')
    args = arg_parser.parse_args()
    config = process_config(args.config)

    agent = SHMAgent(config)
    agent.run()


if __name__ == '__main__':
    main()