# Bugs fixed on `exploration.ipynb`
- The function `is_high_season` had an issue since it didn't consider the time. For example, `is_high_season("2017-12-31 14:55:00")` returned `0` when in reality it should return `1`.
- All calls to `sns.barplot` were missing the `x` and `y` definition.
- To improve visualizations and correctly show the delay rate, the method `get_rate_from_column` was updated. Instead of calculating `rates[name] = round(total / delays[name], 2)`, I think it's best to do `rates[name] = round(100 * delays[name] / total, 2)` to get the ratio of delayed flights to the total number of flights (for a specific column value). This value is now between 0 and 100, where `0` indicates that no flights with that specific column value were delayed and `100` indicates that all flights with that specific column value were delayed. After implementing this change, visualization code had to be updated as well to avoid limiting the y-axis.
- `training_data` was defined but never used. This cell was deleted.
- `top_10_features` included more than 10 features and they weren't the top ones.
- Minor style changes were applied, such as using spaces around operators and removing spaces when defining the value of certain method arguments.
- `xgboost` was not included under installed dependencies.


# Model pick
When comparing the different models' performances, I want to focus on the positive (minority) class since that's the class that represents delays and the model is intended to predict the probability of **delay**. For this, I'll focus on the F1-score for the positive class, since it combines precision and recall into a single metric, providing a balance between the two and accounting for false positives and false negatives. It offers a consolidated evaluation of the model's performance in predicting the positive class while factoring in the class imbalance. Let's review the results for each model:
1. **XGBoost**: 0.00
2. **Logistic Regression**: 0.06
3. **XGBoost with Feature Importance and with Balance**: 0.36
4. **XGBoost with Feature Importance but without Balance**: 0.01
5. **Logistic Regression with Feature Importante and with Balance**: 0.36
6. **Logistic Regression with Feature Importante but without Balance**: 0.03

With this in mind, the best model picked was the third one: **XGBoost with Feature Importance and with Balance**. XGBoost is an ensemble method that can capture complex relationships in the data and it's more suitable to capture non-linear patterns and interactions between features. Plus, XGBoost is highly scalable and effective for large datasets, while LogisticRegression may struggle with very large datasets.