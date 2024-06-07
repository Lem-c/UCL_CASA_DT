import logging


class Logger:
    class CustomFormatter(logging.Formatter):
        COLORS = {
            'DEBUG': '\033[96m',     # Cyan
            'INFO': '\033[92m',      # Green
            'WARNING': '\033[93m',   # Yellow
            'ERROR': '\033[91m',     # Red
            'CRITICAL': '\033[1;91m' # Bold Red
        }
        RESET = '\033[0m'

        def format(self, record):
            log_fmt = '%(asctime)s - %(name)s - %(levelname)s\n%(message)s'
            color_fmt = self.COLORS.get(record.levelname, self.RESET) + log_fmt + self.RESET
            formatter = logging.Formatter(color_fmt)
            return formatter.format(record)

    def __init__(self, name=__name__):
        self.logger = logging.getLogger(name)
        self._setup()

    def _setup(self):
        handler = logging.StreamHandler()
        handler.setFormatter(self.CustomFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def get_logger(self):
        return self.logger
