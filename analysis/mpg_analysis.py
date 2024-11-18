# analysis/mpg_analysis.py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

def load_mpg_data():
    """Load the MPG dataset and clean it by dropping missing values."""
    return sns.load_dataset('mpg').dropna()

def pairs_plot(data):
    """
    Create a pair plot to visualize relationships between features.
    Annotations will be added below the plot for clarity.
    """
    pairplot = sns.pairplot(data)
    plt.show()

    # Save or print relationship details for clarity
    print("\n### Pair Plot Explanation ###")
    print("Diagonal: Histograms of individual features")
    print("Off-diagonal: Scatter plots showing pairwise relationships.")
    print("Examples: ")
    print("1. mpg vs displacement: Indicates a negative relationship.")
    print("2. mpg vs weight: Indicates a negative relationship.")
    print("3. weight vs horsepower: Indicates a positive relationship.")

def linear_regression_mpg_disp(data):
    """
    Perform a simple linear regression on MPG vs displacement and print results.
    """
    X = data[['displacement']]
    y = data['mpg']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

if __name__ == "__main__":
    # Load data and visualize
    data = load_mpg_data()
    pairs_plot(data)

    # Perform regression
    model = linear_regression_mpg_disp(data)
    print("\n### Linear Regression Results ###")
    print("Intercept:", model.params['const'])
    print("Residual Standard Error (RSE):", model.bse['displacement'])
    print("R-squared:", model.rsquared)
    print("F-statistic:", model.fvalue)
    print("P-value:", model.f_pvalue)
