import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from geoshapley import GeoShapleyExplainer
import numpy as np
import cupy as cp

# 设置全局字体和颜色
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['font.size'] = 16

# 读取数据
data = pd.read_excel('预测指标2.xlsx')

# 将数据转换为 GeoDataFrame
data = gpd.GeoDataFrame(
    data, crs="EPSG:32610", geometry=gpd.points_from_xy(x=data.UTM_X, y=data.UTM_Y))

# 选择目标变量和特征
y = data['Nodes importance']
X_coords = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'UTM_X', 'UTM_Y']]

# 数据归一化
scaler = StandardScaler()
X_coords_scaled = scaler.fit_transform(X_coords)
X_coords_scaled = pd.DataFrame(X_coords_scaled, columns=X_coords.columns)  # 将归一化后的数据转换回 DataFrame

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_coords_scaled, y, test_size=0.2, random_state=1)

X_train_gpu = cp.array(X_train)
X_test_gpu = cp.array(X_test)
y_train_gpu = cp.array(y_train)
y_test_gpu = cp.array(y_test)
'''
# 定义XGBoost模型，使用最佳参数并设置GPU参数
best_model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    device='cuda',
    colsample_bytree=0.5,
    gamma=0,
    learning_rate=0.02,
    max_depth=5,
    min_child_weight=1,
    n_estimators=500,
    subsample=0.8
)

# 训练模型
best_model.fit(X_train_gpu.get(), y_train_gpu.get())  # 使用 .get() 将 cupy 数组转换回 numpy 数组
'''
# 定义XGBoost模型
xgb = XGBRegressor(objective='reg:squarederror', random_state=42, device='cuda')

# 定义超参数网格
param_grid = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.02, 0.05],
    'subsample': [0.5, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.7, 0.9, 1],
    'gamma': [0, 0.2, 0.4],
    'min_child_weight': [1, 3, 5, 7],
}

# 5次交叉验证调整超参数
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train_gpu.get(), y_train_gpu.get())

# 最佳模型
best_model = grid_search.best_estimator_

# 模型预测
y_pred_gpu = best_model.predict(X_test_gpu.get())  # 使用 .get() 将 cupy 数组转换回 numpy 数组

# 性能评估
r2 = r2_score(y_test_gpu.get(), y_pred_gpu)
mse = mean_squared_error(y_test_gpu.get(), y_pred_gpu)
mae = mean_absolute_error(y_test_gpu.get(), y_pred_gpu)
relative_errors = np.abs(y_test_gpu.get() - y_pred_gpu) / np.abs(y_test_gpu.get())
mre = np.mean(relative_errors)

# 使用 GeoShapley 进行解释
background_X = X_coords_scaled.values
explainer = GeoShapleyExplainer(best_model.predict, background_X)
rslt = explainer.explain(X_coords_scaled, n_jobs=-1)


# 绘制 SHAP summary plot
rslt.summary_plot(dpi=300)
plt.xlabel('GeoShapley value')
plt.show()

# 绘制部分依赖图
rslt.partial_dependence_plots(figsize=(15, 6), max_cols=2, gam_curve=True)
plt.show()

# 生成总结统计信息
summary_stats = rslt.summary_statistics()
print(summary_stats)
plt.show()

# 检查加性
additivity_check = rslt.check_additivity()
print(additivity_check)

# 输出最佳参数和性能评估到txt文件
with open('summary_bia_XGB_914.txt', 'w') as f:
    f.write("Best parameters found: \n")
    f.write(str(grid_search.best_params_) + "\n\n")
    f.write(f'R2: {r2}\n')
    f.write(f'MSE: {mse}\n')
    f.write(f'MAE: {mae}\n\n')
    f.write(f'MRE: {mre}\n\n')
    f.write("Additivity Check: \n")
    f.write(str(additivity_check) + "\n")
    f.write("Summary Statistics: \n")
    f.write(str(summary_stats) + "\n\n")

# 输出最佳参数
print("Best parameters found: ")
print(grid_search.best_params_)

print(f'R2: {r2}')
print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'MRE: {mre}')


