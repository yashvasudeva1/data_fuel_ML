from imblearn.over_sampling import RandomOverSampler

def random_oversample(X, y, random_state=42):
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled