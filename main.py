# main.py
from analysis.mpg_analysis import load_mpg_data, pairs_plot, linear_regression_mpg_disp
from analysis.iris_analysis import load_and_describe_iris, train_decision_tree
from clustering.kmeans_clustering import kmeans_clustering

if __name__ == "__main__":
    # Task 1: MPG Dataset Analysis
    print("### Task 1: MPG Dataset Analysis ###")
    mpg_data = load_mpg_data()
    pairs_plot(mpg_data)

    mpg_model = linear_regression_mpg_disp(mpg_data)
    print("\n### Linear Regression Results ###")
    print("Intercept:", mpg_model.params['const'])
    print("Residual Standard Error (RSE):", mpg_model.bse['displacement'])
    print("R-squared:", mpg_model.rsquared)
    print("F-statistic:", mpg_model.fvalue)
    print("P-value:", mpg_model.f_pvalue)

    # Task 2: Iris Dataset Analysis
    print("\n### Task 2: Iris Dataset Analysis ###")
    iris_data = load_and_describe_iris()
    iris_model, accuracy = train_decision_tree(iris_data)
    print("\nDecision Tree Accuracy:", accuracy)
    print("Feature Importances:", iris_model.feature_importances_)

    # Task 3: K-means Clustering
    print("\n### Task 3: K-means Clustering ###")
    kmeans_clustering()
