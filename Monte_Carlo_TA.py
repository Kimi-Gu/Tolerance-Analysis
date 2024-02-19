import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, triang, expon

# 定义参数和其分布类型及参数
param1_dist = norm(loc=10, scale=0.5)
param2_dist = uniform(loc=3, scale=2)
param3_dist = triang(c=0.5, loc=2, scale=4)
param4_dist = expon(scale=1)

# 定义模型计算函数
def model(params):
    param1, param2, param3, param4 = params
    return param1 + param2 * param3 - param4

# 设置模拟次数
num_simulations = 1000000

# 生成随机参数值
param1_values = param1_dist.rvs(num_simulations)
param2_values = param2_dist.rvs(num_simulations)
param3_values = param3_dist.rvs(num_simulations)
param4_values = param4_dist.rvs(num_simulations)

# 将参数值组合为参数数组
params = np.array([param1_values, param2_values, param3_values, param4_values])

# 计算模型结果
results = model(params)

# 设置目标范围的上下限
target_min = 12
target_max = 15

# 计算目标范围落在结果中的概率分布
in_target = (results >= target_min) & (results <= target_max)
prob_in_target = np.sum(in_target) / num_simulations

# 绘制输入参数的直方图和分布拟合曲线
plt.subplot(2, 1, 1)
plt.hist(param1_values, bins=30, density=True, alpha=0.7, label='Param1')
plt.hist(param2_values, bins=30, density=True, alpha=0.7, label='Param2')
plt.hist(param3_values, bins=30, density=True, alpha=0.7, label='Param3')
plt.hist(param4_values, bins=30, density=True, alpha=0.7, label='Param4')
plt.xlabel('Parameter Values')
plt.ylabel('Probability Density')
plt.title('Input Parameter Distributions')
plt.legend()

# 绘制结果的直方图和分布拟合曲线
plt.subplot(2, 1, 2)
plt.hist(results, bins=30, density=True, alpha=0.7, label='Results')
plt.xlabel('Results')
plt.ylabel('Probability Density')
plt.title('Monte Carlo Analysis Results')

# 添加分布拟合曲线
x = np.linspace(np.min(results), np.max(results), 100)
param_dist = norm(loc=np.mean(results), scale=np.std(results))
plt.plot(x, param_dist.pdf(x), 'r-', label='Fitted Distribution')

# 绘制目标范围的区域
plt.axvspan(target_min, target_max, alpha=0.3, color='green', label='Target Range')

plt.legend()
plt.tight_layout()
plt.show()

# 打印结果
print("目标范围落在结果中的概率：", prob_in_target*100,"%")
