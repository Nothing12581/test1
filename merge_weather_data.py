import pandas as pd
import os

print("⏳ 1. 构建35040 行基准时间轴 ...")
# 强行生成 2025全年的15分钟标准时间轴，一天 96 个点，绝对是 35040 行！
perfect_time_idx = pd.date_range(start='2025-01-01 00:00:00', end='2025-12-31 23:45:00', freq='15min')
perfect_df = pd.DataFrame({'Time': perfect_time_idx})

# 提取 Month, Day, Hour, Minute 作为万能拉链
for col in ['Month', 'Day', 'Hour', 'Minute']:
    perfect_df[col] = getattr(perfect_df['Time'].dt, col.lower())

print("⏳ 2. 读取并对齐 2025 年气象数据...")
weather_df = pd.read_csv('data/2025_锦州市_history_meteo_data.csv')
selected_weather = ['date', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'shortwave_radiation']
weather_df = weather_df[selected_weather]
weather_df['date'] = pd.to_datetime(weather_df['date'])

for col in ['Month', 'Day', 'Hour', 'Minute']:
    weather_df[col] = getattr(weather_df['date'].dt, col.lower())

# 把气象数据“钉”在完美时间轴上 (丢掉原本粗糙的 date 列)
weather_15m = pd.merge(perfect_df, weather_df.drop(columns=['date']), on=['Month', 'Day', 'Hour', 'Minute'], how='left')

# 气象数据的空白补全：中间的用线性插值，最后缺的那 3 个用前向平推 (ffill)
weather_15m.interpolate(method='linear', inplace=True)
weather_15m.ffill(inplace=True)

print("⏳ 3. 读取并对齐 2013 年负荷数据...")
load_df = pd.read_csv('data/processed/Area1_train.csv')
load_df.rename(columns={'datetime': 'Time', 'load': 'Load'}, inplace=True)
load_df['Time'] = pd.to_datetime(load_df['Time'])
load_df_2013 = load_df[load_df['Time'].dt.year == 2013].copy()

for col in ['Month', 'Day', 'Hour', 'Minute']:
    load_df_2013[col] = getattr(load_df_2013['Time'].dt, col.lower())

print("🔗 4. 缝合...")
# 再把负荷数据“钉”在完美时间轴上
final_df = pd.merge(
    weather_15m,
    load_df_2013[['Load', 'Month', 'Day', 'Hour', 'Minute']],
    on=['Month', 'Day', 'Hour', 'Minute'],
    how='left'
)

# 负荷数据的缺失修复：插值平滑，外加首尾兜底
final_df['Load'] = final_df['Load'].interpolate(method='linear').ffill().bfill()

# 终极清理
final_df.drop(columns=['Month', 'Day', 'Hour', 'Minute'], inplace=True)
cols = ['Time', 'Load', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'shortwave_radiation']
final_df = final_df[cols]

# 保存文件
os.makedirs('data/processed', exist_ok=True)
output_path = 'data/processed/Area1_train_weather_v2.csv'
final_df.to_csv(output_path, index=False)

print(f" 终极大功告成！新数据已保存至: {output_path}")
print(f"   最终合并行数: {len(final_df)} 行 ")