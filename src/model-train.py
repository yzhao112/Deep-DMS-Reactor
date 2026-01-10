import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tabpfn import TabPFNRegressor

# 读取数据
"""
请评委阅读：
下面csv_file地址可自定义，但请注意，如果文件名中包含中文则可能出现报错的情况，若出现报错请将地址替换为英文，并将地址替换为绝对路径（并确保路径中没有中文）。
"""
data = pd.read_csv(r"ASPENV12计算样本（共482个）.csv", encoding='ISO-8859-1')
all_data = data[['T', 'P', 'Catalysis', 'D', 'L']].values
all_labels = data['dms'].values
"""
请评委阅读：
若出现报错显示没有发现'dms'列，请将上面代码中的'dms'替换为数据集中实际的标签列名。例如可能替换为下面注释中的代码。
"""

# all_labels = data['ï»¿dms'].values

# 数据预处理
df = pd.DataFrame(all_data, columns=['T', 'P', 'Catalysis', 'D', 'L'])

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
threshold = 1.5
outliers = df[(df < Q1 - threshold * IQR) | (df > Q3 + threshold * IQR)]
df = df.dropna()
all_data = df.values
all_labels = all_labels[df.index]

# 划分训练集和测试集，比例为8:2
X = all_data
y = all_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train_repeated = np.repeat(X_train, 1, axis=0)
y_train_repeated = np.repeat(y_train, 1, axis=0)

# 初始化TabPFN模型
model = TabPFNRegressor()

# 训练模型
model.fit(X_train_repeated, y_train_repeated)

# 预测
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 计算评估指标
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_corr, _ = pearsonr(y_train, y_pred_train)
test_corr, _ = pearsonr(y_test, y_pred_test)

# 打印评估结果
print(f'训练集 MSE: {train_mse:.4f}')
print(f'测试集 MSE: {test_mse:.4f}')
print(f'训练集 R² Score: {train_r2:.4f}')
print(f'测试集 R² Score: {test_r2:.4f}')
print(f'训练集 相关系数: {train_corr:.4f}')
print(f'测试集 相关系数: {test_corr:.4f}')
# 计算训练集和测试集的预测误差
train_errors = y_pred_train - y_train
test_errors = y_pred_test - y_test

# 计算训练集和测试集误差的标准差
train_std = np.std(train_errors)
test_std = np.std(test_errors)
# 可视化预测结果
plt.figure(figsize=(12, 5))

# 训练集预测结果
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.6)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
x_range = np.linspace(y_train.min(), y_train.max(), 100)
plt.plot(x_range, x_range + 3*train_std, 'g--', alpha=0.7, label='+3σ')
plt.plot(x_range, x_range - 3*train_std, 'g--', alpha=0.7, label='-3σ')
plt.fill_between(x_range, x_range + 3*train_std, x_range - 3*train_std, 
                 alpha=0.1, color='g', label='3σ Confidence Interval')
plt.xlabel('Actual DMS')
plt.ylabel('Predicted DMS')
plt.title(f'Training set (R² = {train_r2:.4f})')

# 测试集预测结果
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
x_range = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(x_range, x_range + 3*test_std, 'g--', alpha=0.7, label='+3σ')
plt.plot(x_range, x_range - 3*test_std, 'g--', alpha=0.7, label='-3σ')
plt.fill_between(x_range, x_range + 3*test_std, x_range - 3*test_std, 
                 alpha=0.1, color='g', label='3σ Confidence Interval')

plt.xlabel('Actual DMS')
plt.ylabel('Predicted DMS')
plt.title(f'Test set (R² = {test_r2:.4f})')

plt.tight_layout()
plt.show()
try:
    # 读取样本数据
    """
    请评委阅读：
    下面csv_file地址可自定义，但请注意，如果文件名中包含中文则可能出现报错的情况，若出现报错请将地址替换为英文，并将地址替换为绝对路径（并确保路径中没有中文）。
    """
    new_data = pd.read_csv('深度学习均匀样本点（共104908个）.csv')
    
    # 提取特征列
    input_features = new_data[['T', 'P', 'Catalysis', 'D', 'L']].values
    
    # 使用模型预测dms值
    dms = model.predict(input_features)
    
    # 将预测结果添加到DataFrame
    new_data['dms'] = dms
    
    # 保存回CSV文件
    new_data.to_csv('深度学习模型预测结果（共104908个）.csv', index=False)
    
    print(f"已成功预测 {len(dms)} 条数据的dms值")
    print(f"结果已保存至 深度学习模型预测结果（共104908个）.csv")
    
except Exception as e:
    print(f"处理外部数据时出错: {e}")
