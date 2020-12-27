# Non-Linear-regression-one-variable

In this project we will implement non linear regression. Our hypothesis is : θ0 + θ1 * x + θ2 * x^2. Note that one feature x is used for non linear combination.

In this project we will use one approaches to solve this problem:

    1) Gradient Descent.
    
We will implement this project with python. 

# Importing libraries and initialization of variables

The number of variables is m = 100, the data are generated randomly. Notice that we have on feature with 100 observations.

    import numpy as np
    import matplotlib.pyplot as plt

    #Variables
    m=100
    x = np.linspace(0, 10, m).reshape((m, 1))
    y = (x +np.random.randn(m, 1))**2   # we use y squared to get a parabolic shape.
    X = np.hstack((np.ones(x.shape),x,x**2 ))
    theta = np.random.rand(3,1)
    itterations = 100
    alpha = 0.0001
    # J : to store at each itteration the cost
    J = np.zeros((itterations))
  



# Defining functions for computations

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
        
# Solution
    thetaSolution = gradientDescent(alpha,itterations,theta)
    Jsol = computeJ(thetaSolution)
    print('\u03B80=', thetaSolution[0], ',\u03B81=',thetaSolution[1],'\u03B82=', thetaSolution[2],'\nJ(\u03B8)=',Jsol)

θ0= [0.21950914] ,θ1= [0.99949936] θ2= [0.9049204] <br/>
J(θ)= 46.2538834525499
    
# Plot of the cost function according to number of itterations

    plt.plot(J_history)
    plt.xlabel('number of itterations')
    plt.ylabel('Cost function J')
    plt.show()
    
![alt text](https://github.com/mohammedAljadd/Non-Linear-regression-one-variable/blob/main/plots/jhist_nonlinear.PNG)

# Plot of fitting line and the data


    plt.plot(X[:,1],h(thetaSolution),label='fitting curve')
    plt.xlabel('x inputs')
    plt.ylabel('y outputs')
    plt.title('Fitting line and data')
    plt.plot(X[:,1], y, 'o',label='training data')
    plt.legend()
    
![alt text](https://github.com/mohammedAljadd/Non-Linear-regression-one-variable/blob/main/plots/fit_nonlinear.PNG)

 # Performance of regression 
 
 ![alt text](https://ashutoshtripathicom.files.wordpress.com/2019/01/rsquarecanva2.png)

 
 This factor should be close to 1.
 
    y_variance = len(y)*np.var(y)
    sum_squared_errors = (2*m)*cost(optimal_beta)
    Performance = 1 - ( sum_squared_errors )/(y_variance)
    print('The performance R is ',Performance) 
    
The performance R is  0.9070555569145956
