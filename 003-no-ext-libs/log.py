import datetime
import time
import atexit
import sys
import os
import pandas as pd
from contextlib import contextmanager


LOG_FILE = 'logs\\{}.log'.format(time.strftime("%Y-%m-%d_%H", time.localtime()))
log_file = open(LOG_FILE, mode='a')
start_time = -1
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def log(*args):
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(time_str, *args)
    print(time_str, *args, file=log_file)
    log_file.flush()


def log_start():
    global start_time
    start_time = time.time()


def log_time(*args):
    log(*args, '[{} sec]'.format(round(time.time() - start_time, 2)))

# @contextmanager
# def log_time(*args):
#     t = time.time()
#     yield
#     log(*args, '[{} sec]'.format(round(time.time() - t, 2)))


def log_trail(char='-', end='\n\n'):
    log(char * 60, '\n\n')


def close_log():
    log_file.close()


atexit.register(close_log)
