# analysis/iris_analysis.py
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def load_and_describe_iris():
    """Load the Iris dataset and display descriptive statistics."""
    iris = sns.load_dataset("iris")
    print("\n### Iris Dataset Descriptive Statistics ###")
    print(iris.describe())
    return iris

def train_decision_tree(data):
    """
    Split data into training and testing sets, train a Decision Tree,
    and compute its accuracy.
    """
    X = data.drop(columns=['species'])
    y = data['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

if __name__ == "__main__":
    # Load and describe dataset
    iris_data = load_and_describe_iris()

    # Train decision tree and evaluate
    model, accuracy = train_decision_tree(iris_data)
    print("\n### Decision Tree Results ###")
    print("Accuracy on Test Dataset:", accuracy)
    print("Feature Importances:", model.feature_importances_)
