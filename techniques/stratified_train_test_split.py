from sklearn.model_selection import train_test_split

def stratified_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )