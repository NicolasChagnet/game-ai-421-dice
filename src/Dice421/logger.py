import logging
import sys

LOGFILE = "game.log"

log = logging.getLogger()
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

fhandler = logging.FileHandler(filename=LOGFILE, mode="a")
fhandler.setFormatter(formatter)
fhandler.setLevel(level=logging.DEBUG)

streamhandler = logging.StreamHandler(stream=sys.stdout)
streamhandler.setFormatter(formatter)
streamhandler.setLevel(level=logging.INFO)


log.addHandler(fhandler)
log.addHandler(streamhandler)


def clear_log():
    with open(LOGFILE, "w") as f:
        f.write("")


def log_enable():
    log.setLevel(logging.DEBUG)


def log_disable():
    log.setLevel(logging.CRITICAL)
