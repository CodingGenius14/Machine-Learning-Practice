import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# loading csv data into a dataframe
df = pd.read_csv("Salary_dataset.csv")

# removing unnecessary columns
df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

# renaming columns for clarity
df.rename(columns={"YearsExperience": "Experience (yrs)", "Salary": "Salary ($)"}, inplace=True)

# setting up the independent and dependent variables
X = df["Experience (yrs)"]
y = df["Salary ($)"]

# train, test, and split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# standardize the independent variable (X)
X_mean = X_train.mean()
X_std = X_train.std()
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std


# plot the relationship between the independent and dependent variables
def plot_data(w, b):
    plt.scatter(X_train, y_train, label="Training Data")
    plt.plot(X_train, w * X_train + b, label="Fitted line", color="red")
    plt.xlabel("Standardized Experience (yrs)")
    plt.ylabel("Salary ($)")
    plt.title("Relationship between Experience and Salary")
    plt.legend()
    plt.show()
    

# gradient descent implementation
def gradient_descent(X, y, iters, lr):
    cost_history = []

    X = X.values
    y = y.values

    w_curr = b_curr = 0
    iterations = iters
    learning_rate = lr
    m = len(X)

    for iter in range(iterations):
        y_predicted = w_curr * X + b_curr
        cost = (1 / (2 * m)) * np.sum((y_predicted - y)**2)
        cost_history.append(cost)
        wd = (1 / m) * np.sum((y_predicted - y) * X)
        bd = (1 / m) * np.sum((y_predicted - y))

        w_curr = w_curr - learning_rate * wd
        b_curr = b_curr - learning_rate * bd
        # print(f"w: {w_curr} | b: {b_curr} | iter: {iter} | cost: {cost}")

    return (w_curr, b_curr, cost_history)


# plot the cost function over time to ensure it is decreasing 
def plot_cost(cost_list):
    plt.plot(cost_list)
    plt.title("Cost Function vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Cost value")
    plt.show()


# store w, b parameters 
(w, b, cost_history) = gradient_descent(X_train, y_train, 1000, 0.019)

# predicted values based on w, b parameters returned by the gradient_descent function
y_pred_test = w * X_test.values + b

# score metrics for model evaluation
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Model predictions on the x-test values:\n{y_pred_test}\n")
print(f"Actual values for the dataset:\n{y_test.values}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

plot_data(w, b)

plot_cost(cost_history)
    

    

