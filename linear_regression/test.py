import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from model import LinearRegression

df = pd.read_csv('./Advertising.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

X = df.drop(['sales'], axis=1)
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


def test_custom_model() -> None:
    print("--- Custom Model Results ---")
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_predict)
    accuracy = 1 - mape
    print(f"Accuracy: {accuracy:.2%}")
    print(f"R2: {r2_score(y_test, y_predict):.2%}\n")


def test_sklearn_model() -> None:
    print("--- Scikit-Learn Model Results ---")
    model_sk = SklearnLinearRegression()
    model_sk.fit(X_train, y_train)

    y_predict_sk = model_sk.predict(X_test)
    mape_sk = mean_absolute_percentage_error(y_test, y_predict_sk)
    accuracy_sk = 1 - mape_sk
    print(f"Accuracy: {accuracy_sk:.2%}")
    print(f"R2: {r2_score(y_test, y_predict_sk):.2%}\n")


if __name__ == "__main__":
    test_custom_model()
    test_sklearn_model()