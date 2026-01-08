from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from model import KNN

df = pd.read_csv('gene_expression.csv')
X = df.drop('Cancer Present',axis = 1)
y = df['Cancer Present']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

if __name__ == '__main__':
    knn = KNN(k=5)
    knn.fit(X_train, y_train)

    accuracy = knn.score(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    new_patient = np.array([4.2, 5.8])
    result = knn.predict([new_patient])
    print("Cancer present" if result[0] == 1 else "Healthy")