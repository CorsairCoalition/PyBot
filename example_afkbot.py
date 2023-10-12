from ggbot.core import PythonBot
import ggbot.utils

class AfkBot(PythonBot):
    """A bot that simply connects to the game and remains stationary."""

    def do_turn(self) -> None:
        pass # do nothing
        

if __name__ == "__main__":
    
    # config = ggbot.utils.get_config_from_file("../config.json")
    config = ggbot.utils.get_config_from_cmdline_args()

    AfkBot().with_config(config).run()