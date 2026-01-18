import numpy as np
import matplotlib.pyplot as plt
from numpy import trapezoid as trapz


def adjoint(func, x_guess, dx, n, relax, target, params=[], RECORD_HIST=False):
    # We first evaluate the error as a function of x

    x_hist = []
    guess_hist = []
    error_hist = []
    error_grad_hist = []
    dx_hist = []

    # We first evaluate the error of the guess
    guess = func(x_guess, *params)

    error = guess - target

    guess_hist.append(guess)
    error_hist.append(error)
    x_hist.append(x_guess)
    error_grad_hist.append(0)

    for k in np.linspace(1, n, num=n, dtype=int):
        # We now timestep in the x space
        guess = func(x_hist[k - 1] + dx, *params)

        error = guess - target

        guess_hist.append(guess)
        error_hist.append(error)
        x_hist.append(x_hist[k - 1] + dx)
        dx_hist.append(dx)
        # Now we computed the gradient

        error_grad = (error_hist[k] - error_hist[k - 1]) / (x_hist[k] - x_hist[k - 1])

        error_grad_hist.append(error_grad)
        if abs(error) < 1e-5:
            break

        # based on this gradient, we can thus find the new x-step based on this gradient # noqa: E501

        dx = -(1 / error_grad) * error_hist[k] * relax

    if RECORD_HIST is True:
        plt.plot(x_hist, error_hist)

        plt.show()

        print("Guess History")
        print(guess_hist)
        print("Error History")
        print(error_hist)
        print("x History")
        print(x_hist)
        print("Error Grad History")
        print(error_grad_hist)
        print("Dx hist")
        print(dx_hist)

    return x_hist[k]


def integrator(func, x_start, x_end, n, params=[]):
    # This function integrates a given function within a given range

    x_array = np.linspace(x_start, x_end, n)

    int_values = func(x_array, *params)

    # We then do trapezoidal integration
    integral = trapz(int_values, x_array)

    return integral
