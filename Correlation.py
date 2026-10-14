import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 输入你的相关系数矩阵
data = {
    'Guangdong': [1.00, 0.31, 0.22, 0.24, 0.25, 0.00],
    'Dynagreen': [0.31, 1.00, 0.07, 0.21, 0.13, 0.09],
    'Power Assets': [0.22, 0.07, 1.00, 0.24, 0.56, 0.11],
    'CR Power': [0.24, 0.21, 0.24, 1.00, 0.33, 0.24],
    'CLP': [0.25, 0.13, 0.56, 0.33, 1.00, 0.16],
    'CATL': [0.00, 0.09, 0.11, 0.24, 0.16, 1.00]
}

labels = ['Guangdong', 'Dynagreen', 'Power Assets', 'CR Power', 'CLP', 'CATL']
df = pd.DataFrame(data, index=labels, columns=labels)

# 画 heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df, annot=True, fmt='.2f', cmap='Greens', vmin=0, vmax=1,
            square=True, linewidths=0.5, cbar_kws={'label': 'Correlation'})
plt.title('Correlation Matrix of Six Holdings')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()