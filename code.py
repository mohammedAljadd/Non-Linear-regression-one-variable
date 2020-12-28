import numpy as np
import matplotlib.pyplot as plt

#Variables
m=100
x = np.linspace(0, 10, m).reshape((m, 1))
y = (x +np.random.randn(m, 1))**2
X = np.hstack((np.ones(x.shape),x,x**2 ))
theta = np.random.rand(3,1)
itterations = 100
alpha = 0.0001
# J : to store at each itteration the cost
J = np.zeros((itterations))

# Hypothesis
def h(theta):
    return X.dot(theta)

#Cost function
def computeJ(theta):
    return 1/(2*m)*np.sum(  (np.square( h(theta)-y) ) ) 

#Gradient
def gradient(theta):
    return (1/m)*X.T.dot(h(theta)-y)

#Gradient descent
def gradientDescent(alpha,itterations,theta):
    for i in range(0,itterations):
        J[i] = computeJ(theta)
        theta = theta - alpha*gradient(theta)
    return theta



############################################################################
#Solution :
thetaSolution = gradientDescent(alpha,itterations,theta)
Jsol = computeJ(thetaSolution)
print('\u03B80=', thetaSolution[0], ',\u03B81=',thetaSolution[1],'\u03B82=', thetaSolution[2],'\nJ(\u03B8)=',Jsol)


############################################################################
#Plot cost function :
plt.plot(J)
plt.xlabel('number of itterations')
plt.ylabel('Cost function J')
plt.show()


############################################################################
#Plot fitting curve :
plt.plot(X[:,1],h(thetaSolution),label='fitting curve')
plt.xlabel('x inputs')
plt.ylabel('y outputs')
plt.title('Fitting line and data')
plt.plot(X[:,1], y, 'o',label='training data')
plt.legend()


############################################################################
#Regression performance 
y_variance = len(y)*np.var(y)
sum_squared_errors = (2*m)*computeJ(thetaSolution)
Performance = 1 - ( sum_squared_errors )/(y_variance)
print('The performance R is ',Performance)