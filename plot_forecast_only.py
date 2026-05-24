import os
import numpy as np
import matplotlib.pyplot as plt
from dispatching.rolling_optimization_1 import ModelPredictor
from utils.dataset import LoadDataset

# 设置中文字体（Windows 默认微软雅黑，防止乱码）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def main():
    print("1. 正在加载多维气象测试集数据...")
    test_path = 'data/processed/Area1_WeatherV2_test.csv'

    dataset = LoadDataset(
        data_path=test_path,
        sequence_length=96,
        scaler_type='standard',
        use_time_features=True
    )

    # 截取用于画图的区间（400个点，约4天的连续负荷）
    plot_len = 400
    start_idx = 0

    sequences = dataset.sequences[start_idx:start_idx + plot_len]
    targets_scaled = dataset.targets[start_idx:start_idx + plot_len]
    actuals = dataset.inverse_transform_target(targets_scaled)

    print("2. 正在加载基线模型 (DE-LSTM) 并生成预测曲线...")
    baseline_path = 'checkpoints/de_lstm_baseline/de_lstm_Area1_20260513_191449/best_model.pth'
    baseline_predictor = ModelPredictor(baseline_path, model_type='de_lstm_baseline')
    baseline_preds_scaled = baseline_predictor.predict_batch(sequences)
    baseline_preds = dataset.inverse_transform_target(baseline_preds_scaled)

    print("3. 正在加载提出模型 (LSTM-Transformer) 并生成预测曲线...")
    proposed_path = 'checkpoints/lstm_transformer/lstm_transformer_Area1_20260513_174529/best_model.pth'
    proposed_predictor = ModelPredictor(proposed_path, model_type='lstm_transformer')
    proposed_preds_scaled = proposed_predictor.predict_batch(sequences)
    proposed_preds = dataset.inverse_transform_target(proposed_preds_scaled)

    base_mape = mean_absolute_percentage_error(actuals, baseline_preds)
    prop_mape = mean_absolute_percentage_error(actuals, proposed_preds)

    print("\n4. 正在绘制左右双排预测对比图...")

    # 创建 1行2列 的极宽画布
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    time_axis = np.arange(plot_len)

    # ==================== 左图：未改进算法 ====================
    axes[0].plot(time_axis, actuals, label='实际负荷 (Actual Load)', color='black', linewidth=2, alpha=0.8)
    axes[0].plot(time_axis, baseline_preds, label=f'DE-LSTM 预测 (MAPE: {base_mape:.2f}%)', color='royalblue',
                 linestyle='--', linewidth=1.5)

    axes[0].set_title('未改进算法模型预测结果 (DE-LSTM)', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('时间步 (15分钟/步)', fontsize=12)
    axes[0].set_ylabel('电力负荷 (MW)', fontsize=12)
    axes[0].legend(fontsize=12, loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # ==================== 右图：改进后算法 ====================
    axes[1].plot(time_axis, actuals, label='实际负荷 (Actual Load)', color='black', linewidth=2, alpha=0.8)
    axes[1].plot(time_axis, proposed_preds, label=f'LSTM-Transformer 预测 (MAPE: {prop_mape:.2f}%)', color='crimson',
                 linewidth=2, alpha=0.9)

    axes[1].set_title('改进后算法模型预测结果 (LSTM-Transformer)', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('时间步 (15分钟/步)', fontsize=12)
    axes[1].set_ylabel('电力负荷 (MW)', fontsize=12)
    axes[1].legend(fontsize=12, loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # 调整两张图的间距
    plt.tight_layout()

    # 保存图片
    os.makedirs('dispatching/results', exist_ok=True)
    save_path = 'dispatching/results/side_by_side_prediction.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n左右对比可视化图表已生成并保存至: {save_path}")


if __name__ == '__main__':
    main()