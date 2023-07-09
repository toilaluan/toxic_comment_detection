import pandas as pd
from sklearn.model_selection import train_test_split
import os
train_data_path = 'dataset/raw_data/train.csv'
X_test_path = 'dataset/raw_data/test.csv'
y_test_path = 'dataset/raw_data/test_labels.csv'
save_path = 'dataset/clean_data/'
os.makedirs(save_path, exist_ok=True)

train_data = pd.read_csv(train_data_path)
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)
test_data = pd.merge(X_test, y_test, on='id')

X_train, X_val, y_train, y_val = train_test_split(train_data.comment_text, train_data.toxic, test_size=0.2)

train_data = pd.concat([X_train, y_train], axis=1)
val_data = pd.concat([X_val, y_val], axis=1)

train_data.to_csv(os.path.join(save_path, 'train.csv'))
val_data.to_csv(os.path.join(save_path, 'val.csv'))
test_data.to_csv(os.path.join(save_path, 'test.csv'))