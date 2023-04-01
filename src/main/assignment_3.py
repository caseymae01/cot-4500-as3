import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

# function: defines the main equation to be used
def function(t: float, w: float):
    return t - (w**2)

# do_work: calls the function to evaluate
def do_work(t, w):
    basic_function_call = function(t, w)
    return basic_function_call

# euler: uses eulers method to calculate the number
def euler():
    # defines the values that you start with
    original_w = 1
    start_of_t, end_of_t = (0, 2)
    num_of_iterations = 10

    # set up h
    h = (end_of_t - start_of_t) / num_of_iterations
    for cur_iteration in range(0, num_of_iterations):

        # defining variable to be used
        t = start_of_t
        w = original_w
        h = h

        # variable that does the inner work in the equation
        inner_math = do_work(t, w)
        next_w = w + (h * inner_math)

        # reassigns variable name
        start_of_t = t + h
        original_w = next_w

    # prints end value
    print("%.5f" % next_w)
    return None

# function2: enters in equation to be used for future references
def function2(t: float, w: float):
    return t - (w**2)

# runge_kutta: calculates value using runge kutta method
def runge_kutta():

    # defines beginning values
    original_w = 1
    start_of_t, end_of_t = (0, 2)
    num_of_iterations = 10

    # set up h
    h = (end_of_t - start_of_t) / num_of_iterations

    # for loop to evaluate using runge kutta math
    for cur_iteration in range(0, num_of_iterations):

        # defining values
        t = start_of_t
        w = original_w
        h = h

        # math to evaluate the number
        first_argument = t + (h / 2)
        another_function_call = function2(t, w)
        second_argument = w + ((h / 2) * another_function_call)
        inner_function1 = function2(first_argument, second_argument)
        third_argument = w + ((h / 2) * inner_function1)
        inner_function2 = function2(first_argument, third_argument)
        fourth_argument = t + h
        fifth_argument = w + (h * inner_function2)
        inner_function3 = function2(fourth_argument, fifth_argument)

        # assigns variable name to the end equation
        next_w = w + (h / 6) * (another_function_call + (2 * inner_function1) + (2 * inner_function2) + inner_function3)

        # reassigns variable names to keep looping through iterations
        start_of_t = t + h
        original_w = next_w

    # prints final value
    print("%.5f" % next_w)
    return None


# gauss_elimination: function to execute gaussian elimination using backwards substitution
def gauss_elimination(A, b):

    # length of matrix to use for ranges
    n = len(b)

    # pushes matrix a and b together
    Ab = np.concatenate((A, b.reshape(n, 1)), axis=1)

    # series of for loops to evaluate through matrices
    for i in range(n):
        for j in range(i+1, n):
            # math inside the main equation
            inside_operation = Ab[j][i]/Ab[i][i]
            for k in range(n + 1):
                # evaluates the new values in the matrix and puts them into the matrix
                Ab[j][k] = Ab[j][k] - inside_operation * Ab[i][k]
    x = Ab[:, n]
    x[n - 1] = Ab[n - 1][n] / Ab[n - 1][n - 1]

    # for loop to determine x values using backwards substitution
    for i in range(n-2, -1, -1):
        x[i] = Ab[i][n]
        for j in range(i+1, n):
            x[i] = x[i] - Ab[i][j]*x[j]
        x[i] = x[i] / Ab[i][i]

    # returns x values in vector form
    return x

# LU_decomposition: finds L and U matrices using decomposition
def LU_decomposition(array):

    # variable for the length of the array and sets up empty matrix for L and U
    n = len(array)
    L = [[0 for x in range(n)]
         for y in range(n)]
    U = [[0 for x in range(n)]
         for y in range(n)]

    # for loops to evaluate L and U matrices
    for i in range(n):
        for k in range(i, n):
            total = 0
            for j in range(i):
                total += L[i][j] * U[j][k]
            U[i][k] = array[i][k] - total

        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                total = 0
                for j in range(i):
                    total += L[k][j] * U[j][i]
                L[k][i] = int((array[k][i] - total) / U[i][i])

    # prints matrices as arrays
    print(np.array(L, dtype=np.double))
    print("")
    print(np.array(U, dtype=np.double))

# diagonally_dominate: defines if the matrix is a diagonally dominate matrix or not
def diagonally_dominate(array):

    # assigns variables to diagonal values and sum of columns
    diagonal_number = np.diag(np.abs(array))
    sum_of_rows = np.sum(np.abs(array), axis=1) - diagonal_number

    # states the conditions for being diagonally dominate and prints accordingly
    if np.all(diagonal_number > sum_of_rows):
        print("True")
    else:
        print("False")

# positive_definite: defines if the matrix is a positive definite
def positive_definite(array):

    # states conditions using eigen values to evaluate as long as it is symmetric, and prints corresponding result
    if np.all(np.linalg.eigvals(array) > 0):
        print("True")
    else:
        print("False")

# main: function that calls previous functions to work
def main():
    # Question 1 Euler method
    euler()
    print("")

    # Question 2 Runge-Kutta method
    runge_kutta()
    print("")

    # Question 3 Gaussian elimination
    A = np.array([[2, -1, 1],
                  [1, 3, 1],
                  [-1, 5, 4]], dtype=np.double)
    b = np.array([6, 0, -3])
    x = gauss_elimination(A, b)
    print(x)
    print("")

    # Question 4 LU factorization
    # a. print the matrix determinant
    array = np.array([[1, 1, 0, 3],
                      [2, 1, -1, 1],
                      [3, -1, -1, 2],
                      [-1, 2, 3, -1]], dtype=np.double)
    determinant = np.linalg.det(array)
    print("%.5f" % determinant)
    print("")

    # b and c. L matrix and U matrix
    LU_decomposition(array)
    print("")

    # Question 5 Diagonally Dominate
    array2 = np.array([[9, 0, 5, 2, 1],
                       [3, 9, 1, 2, 1],
                       [0, 1, 7, 2, 3],
                       [4, 2, 3, 12, 2],
                       [3, 2, 4, 0, 8]], dtype=np.double)

    diagonally_dominate(array2)
    print("")

    # Question 6 Positive Definite
    array3 = np.array([[2, 2, 1],
                       [2, 3, 0],
                       [1, 0, 2]])
    positive_definite(array3)

# calls the function main to execute all code
if __name__ == "__main__":
    main()
