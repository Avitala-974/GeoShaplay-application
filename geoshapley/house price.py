import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from geoshapley import GeoShapleyExplainer
import shap
import contextily as cx

data = pd.read_csv("/home/avitala/PycharmProjects/SHAP/data/seattle_sample_500.csv")

data = gpd.GeoDataFrame(
    data, crs="EPSG:32610", geometry=gpd.points_from_xy(x=data.UTM_X, y=data.UTM_Y))

y = data.log_price

X_coords = data[['bathrooms', 'sqft_living', 'sqft_lot', 'grade', 'condition',
                 'waterfront', 'view', 'age','UTM_X', 'UTM_Y']]

X_train, X_test, y_train, y_test = train_test_split(X_coords, y, random_state=1)

rf_model = RandomForestRegressor(max_depth=10, random_state=0)
rf_model.fit(X_train.values, y_train)

r2_score(y_test, rf_model.predict(X_test))

background_X = X_coords.values
explainer = GeoShapleyExplainer(rf_model.predict, background_X)
rslt = explainer.explain(X_coords,n_jobs=-1)

rslt.summary_plot(dpi=100)
rslt.summary_statistics()
rslt.check_additivity()
rslt.partial_dependence_plots(figsize=(15,6),max_cols=4,gam_curve=True)

coords = data[['UTM_X','UTM_Y']].values
svc = rslt.get_svc(col = [0,1,2,3,4,5,6,7], coef_type="gwr", include_primary=False, coords=coords)

fig, ax = plt.subplots(1, 1,figsize=(10,7),dpi=100)
data.plot(ax=ax,column=rslt.geo,
                  s=10,figsize=(10,10),legend=True)
cx.add_basemap(ax, source=cx.providers.CartoDB.Voyager, crs=data.crs,zoom=11)

fig, ax = plt.subplots(1, 1,figsize=(10,7),dpi=100)
data.plot(ax=ax,column=svc[:,1],
                  s=10,figsize=(10,10),legend=True)
cx.add_basemap(ax, source=cx.providers.CartoDB.Voyager, crs=data.crs,zoom=11)