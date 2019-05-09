import os
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
import yaml
from easydict import EasyDict
from pprint import pprint

from utils.misc import create_dirs


def setup_logging(log_dir):
    log_file_format = '[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d'
    log_console_format = '[%(levelname)s]: %(message)s'

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as config_file:
        try:
            config_dict = yaml.load(config_file)
            config = EasyDict(config_dict)
            return config
        except ValueError:
            print('INVALID YAML file format.. Please provide a good yaml file')
            exit(-1)


def process_config(yaml_file):
    config = get_config_from_yaml(yaml_file)
    print(' The configuration of your experiment is..')
    pprint(config)

    try:
        print(' *************************************** ')
        print(' The experiment mode is {} '.format(config.mode))
        print(' *************************************** ')
    except AttributeError:
        print('ERROR!!..Please provide the exp_name in yaml file..')
        exit(-1)

    # create some important directories to be used for that experiments
    if config.mode == 'pretrain_tnet':
        config.summary_dir = os.path.join('experiments', 'tnet', 'summaries/')
        config.checkpoint_dir = os.path.join('experiments', 'tnet', 'checkpoints/')
        config.out_dir = os.path.join('experiments', 'tnet', 'out/')
        config.log_dir = os.path.join('experiments', 'tnet', 'logs/')
        create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir])
    elif config.mode == 'pretrain_mnet':
        config.summary_dir = os.path.join('experiments', 'mnet', 'summaries/')
        config.checkpoint_dir = os.path.join('experiments', 'mnet', 'checkpoints/')
        config.out_dir = os.path.join('experiments', 'mnet', 'out/')
        config.log_dir = os.path.join('experiments', 'mnet', 'logs/')
        create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir])
    elif config.mode == 'end_to_end':
        config.summary_dir = os.path.join('experiments', 'shm', 'summaries/')
        config.checkpoint_dir = os.path.join('experiments', 'shm', 'checkpoints/')
        config.out_dir = os.path.join('experiments', 'shm', 'out/')
        config.log_dir = os.path.join('experiments', 'shm', 'logs/')
        create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir])
    elif config.mode == 'test':
        config.out_dir = os.path.join('experiments', 'test', 'out/')
        config.log_dir = os.path.join('experiments', 'test', 'logs/')
        create_dirs([config.out_dir, config.log_dir])
    else:
        raise Exception('Please choose a proper experiment mode')

    # setup logging in the project
    setup_logging(config.log_dir)

    logging.getLogger().info('Hi, This is root.')
    logging.getLogger().info('After the configurations are successfully processed and dirs are created.')
    logging.getLogger().info('The pipeline of the project will begin now.')

    return config
