import numpy as np

LEARNING_RATE = 0.001
def fit_GradientDescent(x, t): 
    a, b = np.random.random(2) 
    for iter in range(1000):
        y = a * x + b
        totalError = np.mean((t-y)**2)
        # print(a,b,totalError) 

        gradient_b = np.mean(-2 * (t-y))
        gradient_a = np.mean(-2 * (t-y) * x)
        b = b - LEARNING_RATE * gradient_b
        a = a - LEARNING_RATE * gradient_a
    return a, b

def fit_GradientDescent2(x, t):
    a, b, c = np.random.random(3) 
    for iter in range(10000):
        y = a * x**2 + b*x + c
        totalError = np.mean((t-y)**2)
        # print(a,b,totalError)

        gradient_c = np.mean(-2 * (t-y))
        gradient_b = np.mean(-2 * (t-y) * x)
        gradient_a = np.mean(-2 * (t-y) * x**2)
        c = c - LEARNING_RATE * gradient_c
        b = b - LEARNING_RATE * gradient_b
        a = a - LEARNING_RATE * gradient_a
    return a, b, c

if __name__ == '__main__':
    x = np.array([-2,-1,2,3])
    t = np.array([1,0,3,5]) 

    print(fit_GradientDescent(x,t))
    print(fit_GradientDescent2(x,t))






