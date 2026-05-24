import pandas as pd
from data.data_preprocess import split_train_val_test

print("读取多气象全量数据...")
full_v2_data = pd.read_csv('data/processed/Area1_train_weather_v2.csv')

# 适配预处理函数的列名要求
full_v2_data.rename(columns={'Time': 'datetime'}, inplace=True)

print("调用工具进行 70/15/15 标准切分...")
# 传入 'Area1_WeatherV2'，会自动生成对应的 train/val/test 文件
train_df, val_df, test_df = split_train_val_test(full_v2_data, 'Area1_WeatherV2')