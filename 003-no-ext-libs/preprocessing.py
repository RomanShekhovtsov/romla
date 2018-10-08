from sklearn.preprocessing import StandardScaler
import numpy as np

from log import *
from metrics import *

ONEHOT_MAX_UNIQUE_VALUES = 20
MAX_DATASET_COLUMNS = 1000
BIG_DATASET_SIZE = 500 * 1024 * 1024  # 300MB

empty_date = datetime.datetime.strptime( str(datetime.MINYEAR).rjust(4,'0') + '-01-01', '%Y-%m-%d')
metrics = get_metrics()


def preprocessing(args, model_config):

    with time_metric('read dataset'):
        df = read_csv(args.train_csv, args.nrows)
    # metrics['read_csv'] = time.time() - t
    initial_dataset_size = sys.getsizeof(df)
    is_big = initial_dataset_size > BIG_DATASET_SIZE
    model_config['is_big'] = is_big

    metrics['df_rows'] = df.shape[0]
    metrics['df_cols'] = df.shape[1]
    metrics['df_size'] = initial_dataset_size

    with time_metric('optimize dataframe'):
        optimize_dataframe(df)

    # missing values
    model_config['missing'] = False
    with time_metric('impute missing values'):
        if df.isnull().values.any():
            model_config['missing'] = True
            df.fillna(-1, inplace=True)
        else:
            log('dataset has no missing values')

    df_y = df.target
    df_X = df.drop('target', axis=1)
    df = None

    train_rows, train_cols = df_X.shape
    if train_rows < 2:
        raise Exception('TRAIN SIZE {} < 2.'.format(train_rows))

    metrics['remove low correlated features'] = 0
    metrics['process datetime features'] = 0

    # if is_big:
    #    remove_low_correlated_features(train_cols, initial_dataset_size, df_y, df_X)
    #
    # else:

    # features from datetime
    with time_metric('process datetime features'):
        df_dates = transform_datetime_features(df_X)
        log('features from datetime ({} columns)'.format(len(df_dates.columns)))

        # optimize
        optimize_dataframe(df_dates)

        # missing values
        if df_dates.isnull().values.any():
            model_config['missing_dates'] = True
            df_dates.fillna(-1, inplace=True)
        else:
            log('no missing values in datetime features')

        df_X = pd.concat((df_X, df_dates), axis=1)
        df_dates = None

    # calculate unique values
    with time_metric('process categorical features'):
        df_unique = df_X.apply(lambda x: x.nunique())
        df_const = df_unique[df_unique == 1]
        # df_unique = df_unique[df_unique > 2]
        # df_unique = df_unique[df_unique <= ONEHOT_MAX_UNIQUE_VALUES]
        # df_unique.sort_values(inplace=True)

        # drop constant features
        df_X.drop(df_const.index, axis=1, inplace=True)
        log('{} constant features dropped'.format(df_const.shape[0]))

    df_X, categorical_values = transform_categorical_features(df_X)
    # df_X, categorical_values = onehot_categorical_features(is_big, df_unique, df_X)
    model_config['categorical_values'] = categorical_values

    # use only numeric columns
    used_columns = [
        col_name
        for col_name in df_X.columns
        if check_column_name(col_name) or col_name in categorical_values
    ]

    df_X = df_X[used_columns]
    if len(df_X.columns) == 0:
        raise Exception('ALL FEATURES DROPPED, STOPPING')

    metrics['X_columns'] = len(used_columns)
    model_config['used_columns'] = used_columns
    log('used columns: {}, size: {}'.format(len(used_columns), sys.getsizeof(df_X)))

    with time_metric('impute missing values before scale'):
        if df_X.isnull().values.any():
            model_config['missing'] = True
            df_X.fillna(-1, inplace=True)

    #scaling
    #X = scaling(df_X)

    return df_X, df_y


def check_column_name(name):
    if name == 'line_id':
        return False
    if name.startswith('datetime'):
        return False
    if name.startswith('string'):
        return False
    if name.startswith('id'):
        return False

    return True

def transform_categorical_features(df, categorical_values={}):
    # categorical encoding
    for col_name in list(df.columns):
        if col_name not in categorical_values:
            if col_name.startswith('id') or col_name.startswith('string'):
                categorical_values[col_name] = df[col_name].value_counts().to_dict()

        if col_name in categorical_values:
            col_unique_values = df[col_name].unique()
            for unique_value in col_unique_values:
                df.loc[df[col_name] == unique_value, col_name] = categorical_values[col_name].get(unique_value, -1)

    return df, categorical_values


# categorical encoding
def onehot_categorical_features(is_big, df_unique,df_X):
    categorical_values = dict()

    if not is_big:
        df_cat = pd.DataFrame()
        for col_name in df_unique.index:
            if df_X.shape[1] + df_cat.shape[1] + df_unique[col_name] > MAX_DATASET_COLUMNS:
                break

            col_unique_values = df_X[col_name].unique()
            categorical_values[col_name] = col_unique_values
            for unique_value in col_unique_values:
                df_cat['onehot_{}={}'.format(col_name, unique_value)] = (df_X[col_name] == unique_value).astype(int)

        log('categorical encoding ({} columns)'.format(len(df_cat.columns)))
        optimize_dataframe(df_cat)
        df_X = pd.concat((df_X, df_cat), axis=1)
        df_cat = None

    return df_X, categorical_values


# scaling
def scaling(df_X, model_config):
    with time_metric('scale'):
        scaler = StandardScaler()
        X = scaler.fit_transform(df_X.values.astype(np.float16))
        df_X = None
        log('scale (X size: {})'.format(sys.getsizeof(X)))
        model_config['scaler'] = scaler
        return X


# remove low correlated features
def remove_low_correlated_features(train_cols, initial_dataset_size, df_y, df_X):
    with time_metric('remove low correlated features'):
        new_feature_count = min(train_cols,
                                int(train_cols / (initial_dataset_size / BIG_DATASET_SIZE)))
        # take only high correlated features
        correlations = np.abs([
            np.corrcoef(df_y, df_X[col_name])[0, 1]
            for col_name in df_X.columns if col_name.startswith('number')
        ])
        new_columns = df_X.columns[np.argsort(correlations)[-new_feature_count:]]
        df_X = df_X[new_columns]
        log('remove {} low correlated features'.format(train_cols - new_feature_count))


def parse_dt(x):
    if not isinstance(x, str):
        return empty_date
    elif len(x) == len('2010-01-01'):
        return datetime.datetime.strptime(x, '%Y-%m-%d')
    elif len(x) == len('2010-01-01 10:10:10'):
        return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    else:
        return empty_date


def transform_datetime_features(df):
    datetime_columns = [
        col_name
        for col_name in df.columns
        if col_name.startswith('datetime')
    ]

    df_dates = pd.DataFrame()
    for col_name in datetime_columns:
        df[col_name] = df[col_name].apply(lambda x: parse_dt(x))
        df_dates['number_year_{}'.format(col_name)] = df[col_name].apply(lambda x: x.year)
        df_dates['number_weekday_{}'.format(col_name)] = df[col_name].apply(lambda x: x.weekday())
        df_dates['number_month_{}'.format(col_name)] = df[col_name].apply(lambda x: x.month)
        df_dates['number_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.day)
        df_dates['number_hour_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour)
        #df_dates['number_hour_of_week_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour + x.weekday() * 24)
        #df_dates['number_minute_of_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.minute + x.hour * 60)

    return df_dates