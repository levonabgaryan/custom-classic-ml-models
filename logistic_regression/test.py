import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from model import LogisticRegressionCustom

df = pd.read_csv('gene_expression.csv')
x = df.drop(['Cancer Present'],axis = 1)
y = df['Cancer Present']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101)


def print_metrics(y_true, y_predict, model_name: str):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_predict),
        "Precision": precision_score(y_true, y_predict),
        "Recall": recall_score(y_true, y_predict),
        "F1-Score": f1_score(y_true, y_predict)
    }
    print(f"\n--- {model_name} Evaluation Metrics ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4%}")

def test_custom_model() -> None:
    model = LogisticRegressionCustom()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    print_metrics(y_test, y_predict, "CUSTOM MODEL")


def test_sklearn_model() -> None:
    sklearn_model = SklearnLogisticRegression()
    sklearn_model.fit(x_train, y_train)
    y_predict_sklearn = sklearn_model.predict(x_test)
    print_metrics(y_test, y_predict_sklearn, "SCIKIT-LEARN")

if __name__ == '__main__':
    test_custom_model()
    test_sklearn_model()