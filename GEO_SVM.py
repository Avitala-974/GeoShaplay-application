import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import contextily as cx
from geoshapley import GeoShapleyExplainer
import numpy as np


# 读取数据
data = pd.read_excel('预测指标.xls', sheet_name='all')
# 输出列名以检查是否存在目标列
print("Data columns:", data.columns)

# 将数据转换为 GeoDataFrame
data = gpd.GeoDataFrame(
    data, crs="EPSG:32610", geometry=gpd.points_from_xy(x=data.UTM_X, y=data.UTM_Y))

# 选择目标变量和特征
y = data['Nodes importance']
X_coords = data[['Economic urbanization', 'Population urbanization', 'Spatial urbanization', 'Consumption urbanization', 'Environmental investment',
                 'R&D investment', 'UTM_X', 'UTM_Y']]

# 数据归一化
scaler = StandardScaler()
X_coords_scaled = scaler.fit_transform(X_coords)
X_coords_scaled = pd.DataFrame(X_coords_scaled, columns=X_coords.columns)  # 将归一化后的数据转换回 DataFrame

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_coords_scaled, y, test_size=0.2, random_state=1)

# 定义支持向量机模型
svr = SVR()

# 定义超参数网格
param_grid = {
    'C': [1e-3, 1e-2, 1e-1, 1, 10, 50],
    'gamma': [0.0001, 0.001, 0.1, 1, 5, 10],
    'kernel': ['rbf']
}


# 10次交叉验证调整超参数，设置verbose=2以查看进度
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=10, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 最佳模型
best_model = grid_search.best_estimator_

# 模型预测
y_pred = best_model.predict(X_test)

# 性能评估
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# 使用 GeoShapley 进行解释
background_X = X_coords_scaled.values
explainer = GeoShapleyExplainer(best_model.predict, background_X)
rslt = explainer.explain(X_coords_scaled, n_jobs=-1)

# 绘制 SHAP summary plot
rslt.summary_plot(dpi=300)
plt.show()

# 生成总结统计信息
summary_stats = rslt.summary_statistics()
print(summary_stats)

# 检查加性
additivity_check = rslt.check_additivity()
print(additivity_check)

# 绘制部分依赖图
rslt.partial_dependence_plots(figsize=(15, 6), max_cols=3, gam_curve=True)
plt.show()

# 输出最佳参数和性能评估到txt文件
with open('summary_statistics_svm2022.txt', 'w') as f:
    f.write("Best parameters found: \n")
    f.write(str(grid_search.best_params_) + "\n\n")
    f.write(f'R2: {r2}\n')
    f.write(f'MSE: {mse}\n')
    f.write(f'RMSE: {rmse}\n\n')
    f.write("Summary Statistics: \n")
    f.write(str(summary_stats) + "\n\n")
    f.write("Additivity Check: \n")
    f.write(str(additivity_check) + "\n")

# 输出最佳参数
print("Best parameters found: ")
print(grid_search.best_params_)

print(f'R2: {r2}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')