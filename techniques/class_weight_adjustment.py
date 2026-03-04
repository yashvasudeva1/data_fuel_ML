from sklearn.linear_model import LogisticRegression

def train_with_class_weights(X_train, y_train):
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    return model