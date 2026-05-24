import pandas as pd
import numpy as np


def process_weather_data(input_file, output_file):
    print(f"1. 正在加载原始数据: {input_file} ...")
    df = pd.read_csv(input_file)

    # 将 'date' 列转换为标准时间格式，并设为索引
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    print(f"   原始数据时间跨度: {df.index.min()} 到 {df.index.max()}")
    print(f"   原始数据形状: {df.shape} (每小时1条)")

    # ---------------------------------------------------------
    # 第一步：对特征进行科学分类
    # ---------------------------------------------------------

    # 1. 离散/状态量 (不适合线性插值，采用向前填充 ffill)
    categorical_cols = [
        'weather_code', 'cloud_cover', 'cloud_cover_low',
        'cloud_cover_mid', 'cloud_cover_high',
        'wind_direction_10m', 'wind_direction_100m'
    ]

    # 2. 累积量 (1小时的总量，拆分到15分钟时需要除以4)
    cumulative_cols = [
        'precipitation', 'rain', 'snowfall'
    ]

    # 3. 连续量 (剩余的所有特征，采用线性插值)
    continuous_cols = [
        col for col in df.columns
        if col not in categorical_cols and col not in cumulative_cols
    ]

    print("\n2. 正在进行 15分钟 (96点/天) 重采样与插值计算 ...")

    # 创建 15分钟 频率的新索引
    df_resampled = pd.DataFrame(
        index=pd.date_range(start=df.index.min(), end=df.index.max(), freq='15min')
    )

    # 处理连续量：线性插值
    df_resampled[continuous_cols] = df[continuous_cols].resample('15min').asfreq().interpolate(method='linear')

    # 处理离散量：向前填充 (保留当前小时的状态，直到下一个小时改变)
    df_resampled[categorical_cols] = df[categorical_cols].resample('15min').ffill()

    # 处理累积量：向前填充后除以 4
    df_resampled[cumulative_cols] = df[cumulative_cols].resample('15min').ffill() / 4.0

    # ---------------------------------------------------------
    # 第二步：生成模型专属时间特征
    # ---------------------------------------------------------
    print("3. 正在生成 quarter_hour (0-95) 隐式时间特征 ...")
    df_resampled['quarter_hour'] = df_resampled.index.hour * 4 + df_resampled.index.minute // 15

    # 重置索引，让 date 重新变回普通列
    df_resampled.reset_index(inplace=True)
    df_resampled.rename(columns={'index': 'date'}, inplace=True)

    # ---------------------------------------------------------
    # 第三步：保存结果
    # ---------------------------------------------------------
    print(f"\n4. 正在保存处理后的高频数据至: {output_file} ...")
    df_resampled.to_csv(output_file, index=False)

    print("==================================================")
    print("✅ 处理完成！")
    print(f"   新数据形状: {df_resampled.shape} (每天 96 条)")
    print("==================================================")


if __name__ == "__main__":
    # 你的原始文件名
    INPUT_CSV = "2026_锦州市_history_meteo_data.csv"
    # 输出的高频文件名
    OUTPUT_CSV = "2026_锦州市_WeatherV2_15min.csv"

    process_weather_data(INPUT_CSV, OUTPUT_CSV)