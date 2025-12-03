import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from geoshapley import GeoShapleyExplainer
import numpy as np


# 设置全局字体和颜色
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['font.size'] = 16

# 读取数据
data = pd.read_excel('预测指标2.xlsx')
# 输出列名以检查是否存在目标列
print("Data columns:", data.columns)

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

# 定义随机森林模型，使用指定参数
best_params = {
    'max_depth': 10,
    'max_features': 'log2',
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 500
}
best_model = RandomForestRegressor(**best_params, random_state=42)

# 训练模型
best_model.fit(X_train.values, y_train)
'''
# 定义随机森林模型
rf = RandomForestRegressor(random_state=42)

# 定义超参数网格
param_grid = {
    'n_estimators': [100, 200, 250, 300, 350, 400, 500],
    'max_depth': [10, 20, 30, 35, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
}

# 10次交叉验证调整超参数，设置verbose=2以查看进度
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ")
print(grid_search.best_params_)

# 最佳模型
best_model = grid_search.best_estimator_
'''
# 模型预测
y_pred = best_model.predict(X_test)

# 性能评估
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
relative_errors = np.abs(y_test - y_pred) / np.abs(y_test)
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

# 检查加性
additivity_check = rslt.check_additivity()

# 生成总结统计信息
summary_stats = rslt.summary_statistics()

import contextily as cx  # 新增底图库
def plot_geo_effect(geodata, geo_effect, title="Spatial Effect of Location", save_path="Spatial_Effect_Map.png"):
    """
    绘制位置效应的空间分布图

    参数:
    geodata: 地理数据 (GeoDataFrame)
    geo_effect: 位置效应值数组
    title: 图表标题
    save_path: 图片保存路径
    """
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=150)
    # 确保地理数据和位置效应值长度匹配
    if len(geodata) != len(geo_effect):
        raise ValueError("地理数据和位置效应值长度不匹配")
    # 将位置效应值添加到地理数据
    geodata['geo_effect'] = geo_effect
    # 绘制位置效应图
    geodata.plot(ax=ax, column='geo_effect',
                 cmap='coolwarm', legend=True,
                 markersize=50, alpha=0.8,
                 legend_kwds={'label': "Location Effect",
                              'orientation': "horizontal",
                              'shrink': 0.7})
    # 添加底图
    cx.add_basemap(ax, source=cx.providers.CartoDB.Voyager, crs=geodata.crs, zoom=11)
    # 设置标题和标签
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    # 移除多余的边框
    ax.set_axis_off()
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 绘制位置效应地图
print("绘制位置效应地图...")
plot_geo_effect(data, rslt.geo, title="Spatial Effect of Location on Node Importance")

# 输出最佳参数和性能评估到txt文件
with open('summary_bia_RF_914.txt', 'w') as f:
    f.write("Best parameters found: \n")
    #f.write(str(grid_search.best_params_) + "\n\n")
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
#print(grid_search.best_params_)

print(f'R2: {r2}')
print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'MRE: {mre}')



