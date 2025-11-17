import numpy as np

def gradient_descent(x, y):
    w_curr = b_curr = 0
    iterations = 1000
    learning_rate = 0.08
    m = len(x)

    for iter in range(iterations):
        y_predicted = w_curr * x + b_curr
        cost_function = (1 / m) * sum([val ** 2 for val in (y_predicted - y)])
        wd = (2 / m) * sum((y_predicted - y) * x)
        bd = (2 / m) * sum(y_predicted - y)

        w_curr = w_curr - learning_rate * wd
        b_curr = b_curr - learning_rate * bd

        print(f"w: {w_curr} | b: {b_curr} | iter: {iter} | cost: {cost_function}")

    return (w_curr, b_curr)

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 8, 10, 11, 14])

(w, b) = gradient_descent(x, y)

print(w * 1.5 + b) # predict the value when x = 1.5

