from imblearn.under_sampling import RandomUnderSampler

def random_undersample(X, y, random_state=42):
    rus = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled