from imblearn.ensemble import BalancedRandomForestClassifier

def train_balanced_random_forest(X_train, y_train, random_state=42):
    model = BalancedRandomForestClassifier(
        n_estimators=100,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model