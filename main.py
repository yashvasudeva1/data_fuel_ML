import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def stratified_split(X, y):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    return accuracy, precision, recall, f1


def train_logistic(X_train, y_train, class_weight=None):
    model = LogisticRegression(max_iter=1000, class_weight=class_weight)
    model.fit(X_train, y_train)
    return model


def main():

    X, y = load_dataset("data/Imbalanced_data.csv")

    X_train, X_test, y_train, y_test = stratified_split(X, y)
    X_train, X_test = scale_data(X_train, X_test)

    results = []

    model = train_logistic(X_train, y_train)
    results.append(("Baseline", *evaluate(model, X_test, y_test)))

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    model = train_logistic(X_res, y_res)
    results.append(("Random Oversampling", *evaluate(model, X_test, y_test)))

    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    model = train_logistic(X_res, y_res)
    results.append(("Random Undersampling", *evaluate(model, X_test, y_test)))

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    model = train_logistic(X_res, y_res)
    results.append(("SMOTE", *evaluate(model, X_test, y_test)))

    model = train_logistic(X_train, y_train, class_weight="balanced")
    results.append(("Class Weight Adjustment", *evaluate(model, X_test, y_test)))

    model = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    results.append(("Balanced Random Forest", *evaluate(model, X_test, y_test)))

    results_df = pd.DataFrame(
        results,
        columns=["Technique", "Accuracy", "Precision", "Recall", "F1 Score"]
    )

    print("\nPerformance Comparison on Imbalanced Dataset\n")
    print(results_df)


if __name__ == "__main__":
    main()