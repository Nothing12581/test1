import os
import pandas as pd
import chinese_calendar as conc


def upgrade_dataset(input_file, output_file):
    print(f"正在处理: {input_file} ...")
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件: {input_file}")
        return

    df = pd.read_csv(input_file)

    # 自动识别时间列名
    time_col = None
    for col in ['date', 'datetime', 'Time']:
        if col in df.columns:
            time_col = col
            break

    if not time_col:
        print("❌ 找不到时间列 (date/datetime/Time)！")
        return

    df[time_col] = pd.to_datetime(df[time_col])

    # 1. 拆分 年、月、日、小时
    df['year'] = df[time_col].dt.year
    df['month'] = df[time_col].dt.month
    df['day'] = df[time_col].dt.day
    df['hour'] = df[time_col].dt.hour

    # 2. 判断节假日 (包含周末和中国法定节假日)
    def check_holiday(date_obj):
        try:
            return 1 if conc.is_holiday(date_obj) else 0
        except:
            # 如果年份超出 chinese_calendar 库的支持范围，退化为周末判断
            return 1 if date_obj.weekday() >= 5 else 0

    print("   正在计算节假日特征 (可能需要几秒钟)...")
    df['is_holiday'] = df[time_col].apply(check_holiday)

    df.to_csv(output_file, index=False)
    print(f"✅ 处理完成，已保存至: {output_file}")
    print(f"   当前数据集特征列: {list(df.columns)}\n")


if __name__ == '__main__':
    # 你可以将这里替换为你实际的 训练集/验证集/测试集 路径
    # 这里我们批量处理 data/processed/ 下的核心文件
    base_dir = 'data/processed'

    files_to_process = [
        'Area1_WeatherV2_train.csv',  # 请根据你实际的文件名修改
        'Area1_WeatherV2_val.csv',
        'Area1_WeatherV2_test.csv'
    ]

    for filename in files_to_process:
        input_path = os.path.join(base_dir, filename)
        # 将新文件命名为 V3
        output_path = os.path.join(base_dir, filename.replace('V2', 'V3'))
        upgrade_dataset(input_path, output_path)