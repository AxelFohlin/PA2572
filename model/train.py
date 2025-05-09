import numpy as np
from sklearn.ensemble import RandomForestRegressor

def train_model(df_train, vectorizer):
    tfidf_matrix = vectorizer.fit_transform(df_train['amenities'])
    numerical_features = df_train[['bedrooms', 'bathrooms', 'accommodates', 'minimum_nights', 'maximum_nights', 'longitude', 'latitude']].values

    X_train = np.hstack((tfidf_matrix.toarray(), numerical_features))
    y_train = df_train['price']

    model = RandomForestRegressor(random_state=42, n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', bootstrap=False, max_depth=None)
    model.fit(X_train, y_train)

    return model