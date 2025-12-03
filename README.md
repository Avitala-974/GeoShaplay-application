# GeoShaplay-application
This is the code implementing in practice from a study proposed by Professor Li——GeoShapley: A Game Theory Approach to Measuring Spatial Effects in Machine Learning Models. The algorithm is built on Shapley value and Kernel SHAP estimator.

GeoShapley can explain any model that takes tabular data + spatial features (e.g., coordinates) as the input. Examples of natively supported models include:
1. XGBoost/CatBoost/LightGBM
2. Random Forest
3. MLP or other `scikit-learn` modules.
4. [TabNet](https://github.com/dreamquark-ai/tabnet)
6. [Explainable Boosting Machine](https://github.com/interpretml/interpret)
7. Statistical models: OLS/Gaussian Process/GWR

This project presents the application of GEOshapley combined with XGBoost and RandomForest, employing grid search and 5-fold cross-validation to identify optimal parameters. It also provides code for integrating GEOshapley with SVM and ML; however, due to suboptimal practical performance, these approaches are not discussed in depth.

