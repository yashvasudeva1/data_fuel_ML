from imblearn.over_sampling import SMOTE

def apply_smote(X, y, random_state=42):
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled