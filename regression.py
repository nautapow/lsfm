import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import make_scorer
from sklearn.datasets import make_regression
from sklearn.metrics import make_scorer, r2_score
import pandas as pd
from scipy import stats

def regression_test(x, y):
    """
    Polynomial Regression with LOOCV to evaluate the best degree of polynomial
    use to fit the data.

    Parameters
    ----------
    x : list | 1d array
        Training data
    y : list | 1d array
        Target variable.

    Returns
    -------
    degree in AIC: int,degree in BIC: int, r-squared: 1d-array
        Degree gave least error in each test and r-square for testing prediction.

    """
    X = np.array(x).reshape(-1,1)
    y = np.array(y)
    
    def calculate_metrics(model, X, y, num_params):
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.mean(residuals**2)
        aic = len(y) * np.log(mse) + 2 * num_params
        bic = len(y) * np.log(mse) + num_params * np.log(len(y))
        
        
        return aic, bic
    
    # Try polynomial degrees from 1 to 6
    degrees = np.arange(1, 7)
    
    # Perform polynomial regression for each degree and calculate AIC, BIC, and R-squared with LOOCV
    aic_scores = []
    bic_scores = []
    r2_scores = []
    ytests = []
    ypreds = []
    
    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        
        # Use Leave-One-Out Cross-Validation
        loocv = LeaveOneOut()
        
        # Calculate AIC, BIC, and R-squared scores using LOOCV
        scores = []
        for train_index, test_index in loocv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train, y_train)
            scores.append(calculate_metrics(model, X_test, y_test, degree + 1))
            
            y_pred = model.predict(X_test)
            ytests += list(y_test)
            ypreds += list(y_pred)
        
        rr = r2_score(ytests, ypreds)
        scores_mean = np.mean(scores, axis=0)
        aic_scores.append(scores_mean[0])
        bic_scores.append(scores_mean[1])
        r2_scores.append(rr)
    
    # Plot the results
    plt.plot(degrees, aic_scores, label='AIC')
    plt.plot(degrees, bic_scores, label='BIC')
    plt.plot(degrees, r2_scores, label='R-squared')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Score')
    plt.title('AIC, BIC, and R-squared for Polynomial Regression with LOOCV')
    plt.legend()
    plt.show()
    
    return np.argmin(aic_scores)+1, np.argmin(bic_scores)+1, r2_scores


def regression_poly(x, y, degree=1):
    x = np.array(x)
    X = x.reshape(-1,1)
    y = np.array(y)
    
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    coefficients = model.named_steps['linearregression'].coef_
    intercept = model.named_steps['linearregression'].intercept_
    
    # Generate points for the linear fit curve
    x_fit = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_fit = model.predict(x_fit)
    r_squared = model.score(X, y)
    
    residuals = y - y_fit
    df = len(y) - degree - 1
    mse = np.mean(residuals**2)
    se = np.sqrt(mse / df)
    # Calculate the t-statistic for the hypothesis test
    t_statistic = np.abs(model.named_steps['linearregression'].coef_[0]) / se
    # Calculate the two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(t_statistic, df=df))

# =============================================================================
#     # Plot the x-y scatter plot and linear fit curve
#     plt.scatter(x, y, label='Data Points')
#     plt.plot(x_fit, y_fit, color='red', label=f'Linear Fit: y = {coefficients[1]:.2f}x + {intercept:.2f}')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Polynomial Regression with Degree 1')
#     plt.legend()
#     plt.show()
# =============================================================================
    
    return x_fit, y_fit, round(r_squared,4), p_value