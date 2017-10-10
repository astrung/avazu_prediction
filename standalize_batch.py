import numpy as np
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

def generate_scaler_data(data_frame, scaler):
    list_column_name = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
    list_scaled_column_name = []
    for column_name in list_column_name:
        list_scaled_column_name.append(column_name + '_scaler')

    sub_data = data[list_column_name]
    scaler_data = scaler.transform(sub_data)
    test = pd.DataFrame(scaler_data, columns=list_scaled_column_name)
    result = pd.concat([data_frame, test], axis=1)

    for column_name in list_column_name:
        del result[column_name]
    return result


def fit_chunks(data, scaler):
    data = data.reset_index(drop=True)
    cols = ['C1'] + list(data.loc[:, 'C14':'C21'])
    sub_data = data[cols]
    scaler.partial_fit(sub_data)
    print(scaler.mean_)
    print(scaler.var_)
    print(scaler.n_samples_seen_)
    time.sleep(1)


def write_scaler_csv(write_index, data, path):
    data = data.reset_index(drop=True)
    new_scaler_data = generate_scaler_data(data, scaler)
    if write_index == 0:
        new_scaler_data.to_csv(path, mode='a', index=False, )
    else:
        new_scaler_data.to_csv(path, mode='a', header=False,
                               index=False,)
    time.sleep(1)


scaler = StandardScaler()
training_and_test_fit_index = 0
write_train_csv_index = 0
write_test_csv_index = 0
batch_size = 100000

training_chunks = pd.read_table('C:\\Users\\tuana\\Desktop\\kaggle\\train\\train.csv', chunksize=batch_size, sep=',',
                                index_col=False)
test_chunks = pd.read_table('C:\\Users\\tuana\\Desktop\\kaggle\\test\\test.csv', chunksize=batch_size, sep=',',
                            index_col=False)

for data in training_chunks:
    fit_chunks(data, scaler)

for data in test_chunks:
    fit_chunks(data, scaler)

joblib.dump(scaler, 'standalize.pkl')
training_chunks = pd.read_table('C:\\Users\\tuana\\Desktop\\kaggle\\train\\train.csv', chunksize=batch_size, sep=',',
                                index_col=False, dtype={'id': str})
test_chunks = pd.read_table('C:\\Users\\tuana\\Desktop\\kaggle\\test\\test.csv', chunksize=batch_size, sep=',',
                            index_col=False, dtype={'id': str})

for data in training_chunks:
    print(write_train_csv_index)
    write_scaler_csv(write_train_csv_index, data, "C:\\Users\\tuana\\Desktop\\kaggle\\scale_train.csv")
    write_train_csv_index = write_train_csv_index + 1

for data in test_chunks:
    print(write_test_csv_index)
    write_scaler_csv(write_test_csv_index, data, "C:\\Users\\tuana\\Desktop\\kaggle\\scale_test.csv")
    write_test_csv_index = write_test_csv_index + 1
