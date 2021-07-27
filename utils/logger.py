import logging
from config.solofy import LOGFILENAME


class Logger:
    __instance = None

    @staticmethod
    def get_instance(log_level=logging.info):
        if Logger.__instance == None:
            Logger(log_level)
        return Logger.__instance

    def __init__(self, log_level):
        self.log_level = log_level
        if Logger.__instance != None:
            raise Exception("Logger is a singleton!")
        else:
            Logger.__instance = self

    def get_logger(self):
        logging.basicConfig(
            filename=LOGFILENAME,
            filemode="w",
            format="%(asctime)s,%(msecs)d - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d:%H:%M:%S",
        )
        logger = logging.getLogger()
        logger.setLevel(self.log_level)
        return logger
