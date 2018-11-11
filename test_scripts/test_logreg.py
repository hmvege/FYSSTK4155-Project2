from sklearn.linear_model import    *
from scipy.special import expit
import numpy as np

def log_likelihood(X, y, weights, Lambda):
    p = np.dot(X, beta)
    ll = - np.sum( y*p - np.log(1 + np.exp(p)) ) - 0.5*Lambda*np.dot(weights, weights) # Change sign here
    return ll

def gradientAscent(X,y,Lambda,eta=1e-4,max_iters=150,tolerance=1e-4):
    
    #Initialize beta-parameters
    beta   = np.zeros(X.shape[1])
    norm   = 100
    
    gradient = np.zeros(X.shape[1])
    for i in range(0,max_iters):
        
        z = np.dot(X, beta)
        p = expit(z)

        if i%100 == 0:
            print(i, gradient)

        gradient = np.dot(X.T,y-p) - Lambda*beta #Note the signs in log-likelihood, change sign in last term here

        beta    += eta*gradient #Gradient ascent, change sign to negative for gradient descent
        norm     = np.linalg.norm(gradient)
        
        if(norm < tolerance):
            print("Gradient descent converged to given precision in %d iterations" % i)
            break
    
    return beta, norm



lmbdas = [0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
lmbdas = [1.0]

np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))

from sklearn import datasets

iris = datasets.load_iris()
X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0



np.set_printoptions(precision=5)
print("*******************************************************************")
for Lambda in lmbdas:

    #For some reason C = 1/lambda in sci-kits logistic regression
    if(Lambda > 0):
        C = 1.0/Lambda
    else:
        C = 1e15

    model = LogisticRegression(fit_intercept=True,C=C,tol=1e-8)
    model.fit(X,y)

    intercept   = np.ones((X.shape[0], 1))
    X_intercept = np.hstack((intercept, X))   

    beta, norm = gradientAscent(X_intercept,y,Lambda=Lambda,max_iters=400000,tolerance=1e-4)

    beta_scikit = np.zeros(len(beta))
    beta_scikit[0] = model.intercept_
    beta_scikit[1:] = model.coef_[0]

    print("Reg strength (lambda): ", Lambda)
    print("beta scikit          : ", beta_scikit)
    print("beta gradient ascent : ", beta)
    print("|beta_s - beta_ga|   : ", abs(beta-beta_scikit))
    print("*******************************************************************")