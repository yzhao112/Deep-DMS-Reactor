import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tabpfn import TabPFNRegressor
import torch

# 设置环境变量以同步CUDA操作，便于调试
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA操作，获取更准确的错误堆栈

# 检查CUDA是否可用
if not torch.cuda.is_available():
    raise RuntimeError("CUDA不可用，无法继续。请确保GPU驱动和CUDA已正确安装。")

# 读取数据
"""
请评委阅读：
下面csv_file地址可自定义，但请注意，如果文件名中包含中文则可能出现报错的情况，若出现报错请将地址替换为英文，并将地址替换为绝对路径（并确保路径中没有中文）。
"""
data = pd.read_csv(r"ASPENV12计算样本（共482个）.csv", encoding='ISO-8859-1')
all_data = data[['T', 'P', 'Catalysis', 'D', 'L']].values
all_labels = data['dms'].values

# 创建DataFrame并处理异常值
df = pd.DataFrame(all_data, columns=['T', 'P', 'Catalysis', 'D', 'L'])
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
threshold = 1.5
outliers = df[(df < Q1 - threshold * IQR) | (df > Q3 + threshold * IQR)]
df = df.dropna()

# 更新数据和标签
all_data = df.values
all_labels = all_labels[df.index]

# 划分训练集和测试集
X = all_data
y = all_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 复制训练数据（可选）
X_train_repeated = np.repeat(X_train, 1, axis=0)
y_train_repeated = np.repeat(y_train, 1, axis=0)

# 初始化TabPFN模型，强制使用CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 检查TabPFN版本
try:
    import tabpfn
    version = tabpfn.__version__
    print(f"TabPFN版本: {version}")
    if version < '0.1.8':
        print("警告: 当前TabPFN版本可能不支持所有参数。建议升级到0.1.8或更高版本。")
except:
    print("无法获取TabPFN版本信息")

# 移除不支持的参数
model = TabPFNRegressor(device=device)

# 训练模型
try:
    print("开始训练模型...")
    model.fit(X_train_repeated, y_train_repeated)
    print("模型训练完成")
except RuntimeError as e:
    if "CUDA error: invalid configuration argument" in str(e):
        print("CUDA配置参数错误。尝试减少集成配置数或使用CPU。")
        # 回退到CPU
        model = TabPFNRegressor(device='cpu')
        model.fit(X_train_repeated, y_train_repeated)
    else:
        raise e

# 分批预测函数
def predict_in_batches(model, X, batch_size=32):
    predictions = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        try:
            batch_preds = model.predict(batch)
            predictions.append(batch_preds)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"批次 {i}-{i+batch_size} 内存不足，尝试更小的批次")
                # 尝试更小的批次
                sub_batch_size = max(1, batch_size // 2)
                sub_predictions = []
                for j in range(0, len(batch), sub_batch_size):
                    sub_batch = batch[j:j+sub_batch_size]
                    sub_predictions.append(model.predict(sub_batch))
                predictions.append(np.concatenate(sub_predictions))
            else:
                raise e
    return np.concatenate(predictions)

# 预测
print("开始预测训练集...")
y_pred_train = predict_in_batches(model, X_train, batch_size=32)
print("开始预测测试集...")
y_pred_test = predict_in_batches(model, X_test, batch_size=32)

# 评估模型
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print(f"测试集MSE: {mse:.4f}")
print(f"测试集R²: {r2:.4f}")

try:
    # 读取外部数据
    print("读取外部数据...")
    """
    请评委阅读：
    下面csv_file地址可自定义，但请注意，如果文件名中包含中文则可能出现报错的情况，若出现报错请将地址替换为英文。
    """
    new_data = pd.read_csv('深度学习均匀样本点（共104908个）.csv')
    
    # 提取特征列
    input_features = new_data[['T', 'P', 'Catalysis', 'D', 'L']].values
    
    # 使用模型预测dms值（分批）
    print("开始预测外部数据...")
    dms = predict_in_batches(model, input_features, batch_size=32)
    
    # 将预测结果添加到DataFrame
    new_data['dms'] = dms
    
    # 保存回CSV文件
    """
    请评委阅读：
    下面csv_file地址可自定义，但请注意，如果文件名中包含中文则可能出现报错的情况，若出现报错请将地址替换为英文。
    """
    new_data.to_csv('深度学习模型预测结果（共104908个）.csv', index=False)
    
    print(f"已成功预测 {len(dms)} 条数据的dms值")
    print(f"结果已保存至 深度学习模型预测结果（共104908个）.csv")
    
except Exception as e:
    print(f"处理外部数据时出错: {e}")
