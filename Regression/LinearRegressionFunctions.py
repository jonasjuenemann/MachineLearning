# Gradient Descent for Intercept, bestimmt den "Fehler-Wert" fuer einen bestimmten Intercept b

def get_gradient_at_b(x, y, b, m):
    diff = 0
    N = len(x)
    for i in range(N):
        diff += (y[i] - (m * x[i] + b))
    b_gradient = (-2 / N) * diff
    return b_gradient


# Gradient Descent for Slope

def get_gradient_at_m(x, y, b, m):
    diff = 0
    N = len(x)
    for i in range(N):
        diff += x[i] * (y[i] - (m * x[i] + b))
    m_gradient = (-2 / N) * diff
    return m_gradient


def step_gradient(b_current, m_current, x, y, learning_rate):
    b_gradient = get_gradient_at_b(x, y, b_current, m_current)
    m_gradient = get_gradient_at_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]


def gradient_descent(x, y, learning_rate, num_iterations):
    b = 0
    m = 0
    for i in range(num_iterations):
        b, m = step_gradient(b, m, x, y, learning_rate)
    return [b, m]
