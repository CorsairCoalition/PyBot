import randombot
from pythonbot import RedisConnectionManager

if __name__ == "__main__":
    with RedisConnectionManager() as rcm:
        rcm.register(randombot.RandomBot("cortex-7LQqyM8"))
        rcm.run()