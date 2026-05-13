import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns


def load_results(result_dir):
    """从指定的实验结果目录中加载预测数据"""
    experiments = {}
    for file in os.listdir(result_dir):
        if file.endswith('_results.json') and file != 'comparison_results.json':
            exp_name = file.replace('_results.json', '')
            with open(os.path.join(result_dir, file), 'r', encoding='utf-8') as f:
                experiments[exp_name] = json.load(f)
    return experiments


def plot_prediction_comparisons(result_dir):
    experiments = load_results(result_dir)
    if not experiments:
        print("未找到实验结果文件！")
        return

    # 设置中文字体（根据你的系统可能需要调整）
        # 设置中文字体（加入微软雅黑作为首选，防止黑体找不到）
    plt.style.use('seaborn-v0_8')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号（防止负号变成小方块）


    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('负荷预测模型性能对比', fontsize=18, fontweight='bold',y=1.02)

    colors = {'baseline': 'blue', 'proposed': 'red', 'ablation': 'green'}

    # 获取实际值 (假设所有实验的 actuals 是一样的)
    first_exp = list(experiments.values())[0]
    actuals = first_exp.get('actuals', [])
    steps = range(len(actuals))

    # 1. 画折线图：实际负荷 vs 各模型预测负荷
    axes[0].plot(steps, actuals, 'k-', alpha=0.7, label='实际负荷 (Actual)', linewidth=2)
    for exp_name, results in experiments.items():
        color = colors.get(exp_name, 'orange')
        axes[0].plot(steps, results['predictions'], color=color,
                     label=f'{exp_name} 预测', alpha=0.8, linestyle='--')

    axes[0].set_title('预测负荷序列对比', fontsize=14)
    axes[0].set_ylabel('负荷 (MW)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 画折线图：各模型预测误差曲线
    for exp_name, results in experiments.items():
        color = colors.get(exp_name, 'orange')
        errors = np.array(results['predictions']) - np.array(results['actuals'])
        axes[1].plot(steps, errors, color=color, label=f'{exp_name} 误差', alpha=0.8)

    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1].set_title('时序预测误差分布', fontsize=14)
    axes[1].set_ylabel('误差 (MW)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 柱状图：MAPE (平均绝对百分比误差) 对比
    mape_values = []
    exp_names = []
    for exp_name, results in experiments.items():
        mape = results.get('overall_metrics', {}).get('mape', 0)
        mape_values.append(mape)
        exp_names.append(exp_name)

    bars = axes[2].bar(exp_names, mape_values, color=['blue', 'red', 'green'][:len(exp_names)])
    axes[2].set_title('预测准确率评估 (MAPE)', fontsize=14)
    axes[2].set_ylabel('MAPE (%)')

    # 在柱子上打上具体数值
    for bar, value in zip(bars, mape_values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(result_dir, 'prediction_comparison_only.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"预测对比图已保存至: {save_path}")


if __name__ == '__main__':
    # 将此处的路径替换为你实际跑出来的带有时间戳的结果文件夹
    TARGET_RESULT_DIR = "results/comparison_Area1_20260426_155839"
    plot_prediction_comparisons(TARGET_RESULT_DIR)