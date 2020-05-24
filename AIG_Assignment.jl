using CSV, DataFrames, Plot, Statistics

#Using dataset
bankDataset = CSV.read("C:\\Users\\jimmy\\Documents\\AIG_Assignment\\bank-additional-full")

#Removing unwanted columns
select!(bankDataset, Not(:contact))
select!(bankDataset, Not(:education))
select!(bankDataset, Not(:default))
select!(bankDataset, Not(:month))
select!(bankDataset, Not(:day_of_week))
select!(bankDataset, Not(:campain))
#removing 

#converting dataset to numeric values
bankDataset = convert(Matrix,bankDataset)

#Choosing your features for X
X = bankDataset[:, 1:15]

#Defining the Y values
Y = bankDataset[:, 16]

X_matrix = convert(Matrix, X)


X_train = X[1:58, :]

X_test = X[59+1:end, :]

Y_train = Y[1:1, :]

Y_test = Y[1+1:end, :]
"""
The Dataset has been cleaned and prepared
"""

"""
Implementing the Regulizized Logical Regression Algorithm
"""
#Sigmoid function
function
    sigmoid(z)
    return 1 ./ (1 .+ exp.(.-z))
end

#regularized cost function
function
    regularised_costFunction(X, y, θ, λ)

#size of training examples
       m = length(y)

#hypothesis of sigmoid function
       h= sigmoid(X * 0)

#when y=1
       yfunctionIsOne = ((1 .- y)' * log.(1 .-h))

#lamba for regulization
       lamba = (λ/(2*m) * sum(θ[2 : end] .^ 2))

#Almost forgot WHEN Y=0
       yfunctionISZero = ((1 .- y)' * log.(1 .- h))

#cOST function of j theta
       jParameterThetaCost = (1/m) * (yFunctionIsOne - yfunctionISZero) + lamba

#minimized cost function of parameter thetaa 0
       rGradientD = (1/m) * (X') * (h-y) + ((1/m) * (λ * θ))

#minimized cost function of parameter thetaa 1 and beyond
       rGradientD[1] = (1/m) * (X[:, 1])' * (h-y)
   return (rGradientD, rGradientD[1])
end

function reg_logistic_regression(X, y, λ, fit_intercept=true, η=0.01, max_iter=1000)

#size of training examples
       m = length(y)

#the fit intercept
       fit_intecept(X)

#n = number of features. used to initialize vector for theetaa
       n = size(X)[15]

#initialized from n
       θ = zeros(n)

#Getting to initialize the cost vector
       rGradientD = zeros(max_iter)

#Simultanious update
       for iter in range(1, stop=max_iter)
          rGradientD[iter], minJ = regularised_cost(X, y, θ, λ)
          θ = θ - (η * minJ)
       end
   return (θ, minJ)
end
