import argparse
import json

def get_config_from_cmdline_args() -> dict:
    """A simple command line argument parser that can load config.json files for your bot
    Use --help at the command line for more info."""
    parser = argparse.ArgumentParser(
        prog = 'GG Framework Python Bot argument parser',
        description = 'A simple command line argument parser that can load config.json files for your bot',
        epilog = 'For more information, see https://CorsairCoalition.github.io')
    parser.add_argument('-c', '--config', dest='config_file', required=True, help='The config.json file used by this bot.')
    parser.add_argument('--debug', action='store_true', dest='do_debug', help='Enables debugging mode on the Python bot, providing a printout of each call to move(). If the caller option is provided, these printouts will include that caller string at the front of the printout.')
    args = parser.parse_args()
    
    config_dict = json.load(open(args.config_file))
    config_dict['GGBOT_DEBUG'] = args.do_debug

    return config_dict

def get_config_from_file(file_path:str) -> dict:
    """Reads the bot configuration from the given file.
    
    Args:
        file_path: The relative path to this bot's configuration file."""
    
    config_dict = json.load(open(file_path))
    return config_dict