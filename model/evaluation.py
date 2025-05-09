import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def evaluate_model(model: RandomForestRegressor, df_test, vectorizer):
    tfidf_matrix = vectorizer.transform(df_test['amenities'])
    numerical_features = df_test[['bedrooms', 'bathrooms', 'accommodates', 'minimum_nights', 'maximum_nights', 'longitude', 'latitude']].values

    X_train = np.hstack((tfidf_matrix.toarray(), numerical_features))
    y_train = df_test['price'].to_numpy()

    y_pred = model.predict(X_train)

    mn = min(y_train.min(), y_pred.min())
    mx = max(y_train.max(), y_pred.max())

    fig, ax = plt.subplots(figsize=(10, 6))
    # Scatter
    ax.scatter(y_train, y_pred, alpha=0.5, label='Listings')

    # Vertical error lines
    for yt, yp in zip(y_train, y_pred):
        ax.plot([yt, yt], [yt, yp], alpha=0.3, linestyle='-')

    # Ideal prediction line (y = x)
    ax.plot([mn, mx], [mn, mx], linestyle='--', linewidth=2, label='Perfect')

    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs. Predicted Prices with Error Lines")
    ax.legend()
    ax.grid(True)

    return {
        "RMSE": root_mean_squared_error(y_train, y_pred),
        "MAE": mean_absolute_error(y_train, y_pred),
        "R2": r2_score(y_train, y_pred),
    }, fig

def display_feature_importance(model, vectorizer):
    tfidf_features = vectorizer.get_feature_names_out()
    numeric_features = ['bedrooms', 'bathrooms', 'accommodates', 'minimum_nights', 'maximum_nights', 'longitude', 'latitude']
    feature_names = list(tfidf_features) + numeric_features

    importances = model.feature_importances_

    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    feature_df = feature_df.sort_values(by='Importance', ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_df['Feature'], feature_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Top 20 Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return feature_df