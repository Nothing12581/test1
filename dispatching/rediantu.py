import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 解决中文显示问题（老规矩）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 2. 伪造一个 96 个步长的注意力权重 (模拟真实的 weights)
# 假设模型把 60% 的注意力放在了昨天同一时刻（第 0 步），20% 放在了刚刚过去的一刻（第 95 步）
weights = np.random.rand(1, 96) * 0.05  # 其他时刻随便给点极小的关注度
weights[0, 0] = 0.60  # 强周期依赖
weights[0, 95] = 0.20 # 强近期依赖

# 3. 召唤复印机：画热力图
plt.figure(figsize=(15, 3))
sns.heatmap(weights, cmap="YlOrRd", cbar_kws={'label': '注意力分配权重'})

plt.title("LSTM-Transformer 自注意力热力图 (模拟效果)", fontsize=16)
plt.xlabel("过去的时间步长 (从 -96 到 当前)", fontsize=12)
plt.yticks([]) # 隐藏 Y 轴刻度，因为只有一行

# 4. 展示图表
plt.show()