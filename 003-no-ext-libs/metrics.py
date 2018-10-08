import pandas as pd
import os
import uuid
import hashlib

from utils import *
from log import *

__metrics = {}

ROOT_METRICS_FOLDER = 'metrics/'
run_id = str(uuid.uuid4())


@contextmanager
def time_metric(name):
    t = time.time()
    yield
    __metrics[name] = time.time() - t
    log(name,'[{} sec]'.format(round(__metrics[name], 2)))


def get_metrics():
    return __metrics

def save_metrics(__metrics, subfolder):

    df = pd.DataFrame(__metrics, index=(1,))
    df['run_id'] = run_id
    columns = ''.join(df.columns)
    folder = 'metrics/' + subfolder + '/'

    if not os.path.exists(folder):
        os.makedirs(folder)

    hash_digest = hashlib.md5(columns.encode('utf-8')).hexdigest()
    file_name = folder + hash_digest.upper() + '.csv'
    file_exists = os.path.exists(file_name)
    df.to_csv(file_name, mode='a', header=not file_exists, index=False, sep=';')
    log('metrics {} to file "{}"'.format('appended' if file_exists else 'saved', file_name))


def load_metrics(subfolder):
    folder = ROOT_METRICS_FOLDER + subfolder + '/'
    if not os.path.exists(folder):
        log('folder "{}" doesn''t exists'.format(folder))
        return None

    files = os.listdir(path=folder)
    log('metrics files:', files)
    df_all = pd.DataFrame()

    for file_name in files:
        df = pd.read_csv(folder + file_name, sep=';')
        df_all = pd.concat((df_all, df), sort=False)

    return df_all
