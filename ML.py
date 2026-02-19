"""Example regressions: linear and polynomial predictions.

This module contains small examples showing how to fit simple
linear and polynomial regression models and produce predictions.
"""

# pylint: disable=invalid-name

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


def simple_linear_regression_example() -> None:
    """Fit a simple linear regression and print a prediction.

    The function plots the sample points but does not show the plot to
    avoid blocking interactive environments.
    """
    heights = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
    weights = [8, 10, 12, 14, 16, 18, 20]
    plt.scatter(heights, weights, color="black")
    plt.xlabel("height")
    plt.ylabel("weight")

    reg = LinearRegression()
    reg.fit(heights, weights)
    x_height = [[12.0]]
    prediction = reg.predict(x_height)
    print(f"Simple linear regression prediction for {x_height}: {prediction}")


def train_test_split_example() -> None:
    """Demonstrate train/test split, fit and score a model."""
    X = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
    y = [8, 10, 12, 14, 16, 18, 20]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7
    )
    print("Training Features", x_train)
    print("Training Labels", y_train)
    print("Testing Features", x_test)
    print("Testing Labels", y_test)

    reg = LinearRegression()
    reg.fit(x_train, y_train)
    score = reg.score(x_test, y_test)
    print(f"Accuracy - test set: {score * 100.0:.2f}%")
    predictions = reg.predict(x_test)
    print(f"Predictions on test set: {predictions}")


def polynomial_regression_example() -> None:
    """Fit a polynomial (via pipeline) and print predictions."""
    x = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
    y = [16, 25, 36, 49, 64, 81, 100]

    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    print(f"Linear regression prediction for 11: {lin_reg.predict([[11]])}")

    polynomial_regression = make_pipeline(
        PolynomialFeatures(degree=1, include_bias=False), LinearRegression()
    )
    polynomial_regression.fit(x, y)
    x_height = [[20.0]]
    target_predicted = polynomial_regression.predict(x_height)
    print(f"Polynomial regression prediction for {x_height}: {target_predicted}")


def main() -> None:
    """Run all examples."""
    simple_linear_regression_example()
    print("=====================================================================")
    train_test_split_example()
    print("=====================================================================")
    polynomial_regression_example()


if __name__ == "__main__":
    main()