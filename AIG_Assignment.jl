using CSV, DataFrames
"""
Data cleaning and preparation done by both Jimmy and Janny
"""
#Using dataset
bankDataset = CSV.read("C:\\Users\\jimmy\\Documents\\AIG_Assignment\\bank-additional-full.csv")

#Removing unwanted columns
select!(bankDataset, Not(:contact))
select!(bankDataset, Not(:education))
select!(bankDataset, Not(:default))
select!(bankDataset, Not(:month))
select!(bankDataset, Not(:day_of_week))
select!(bankDataset, Not(:campaign))

#converting dataset to numeric values
bankDataset = convert(Matrix,bankDataset)

#Choosing your features for X
X = bankDataset[:, 1:end-1]

#Defining the Y values
Y = bankDataset[:, end]

X_matrix = convert(Matrix, X)


X_train = X[1:58, :]
X_test = X[58+1:end, :]

Y_train = Y[1:58, :]
Y_test = Y[58+1:end, :]
"""
The Dataset has been cleaned and prepared
"""

"""
Done by Janny David
"""
"""
Implementing the Regulizized Logical Regression Algorithm
"""
#hypothesis for Logistic Regression with sigmoid function
function hypothesis(x, θ)
       z = -θ' * X
   return 1 / (1 + exp(z))
end

#regularized cost function
function
    regularised_costFunction(X, Y, θ, λ)

#size of training examples
       m = length(Y)

#hypothesis of sigmoid function
       hypothesis = sigmoid(X * 0)

#when y=1
       yfunctionIsOne = ((- Y)' * log.(1 .- hypothesis))

#when Y=0
       yfunctionIsZero = ((1 .- Y)' * log.(1 .- hypothesis))

#lamba for regulization
       regulationTerm = (λ/(2*m) * sum(θ[2 : end] .^ 2))

#Cost function
       J(θ) = (1/m) * (yFunctionIsOne - yfunctionISZero) + regulationTerm

       #minimized cost function gradient descent for theeta 0
       θ0 = (1/m) * (X[:, 1])' * (hypothesis - Y)
       #minimized cost function gradient descent for theeta 1 till theetaa n
       θj[1] = (1/m) * (X') * (hypothesis - Y) + ((1/m) * (λ * θ))
    return (θ0, θj[1])
end

#iterations for gradient descent
function logistic_reg(X, Y, λ, fit_intercept=true, η=0.01, iteration = 1000)

    #Size of training examples
    m = length(Y);

    if fit_intercept
        constant = ones(m, 1)
        X = hcat(constant, X)
    else
        X
    end
    #n features to initialize vectors and using the iteration to initialize J
    n = size(X)[2]
    θ = zeros(n)
    J = zeros(iteration)

    for w in range(1, stop = iteration)

        J[w], θj = regularised_costFunction(X, y, θ, λ)
        θ = θ - (η * θj)
    end

    return (θ, J)
end

"""
Done by Jimmy Damiao
"""
"""
Classifiers for dataset
"""

"""
Accuracy
"""
function predict_probability(X, θ, fit_intercept = true)
    m = size(X)[1]

    if fit_intercept
        constant = ones(m, 1)
        X = con(constant, X)
    elseif fit_intercept!
        X #when fit is not specified
    hypothesis = sigmoid(X * θ)
    return hypothesis
end
end

#Binary classifier
function predictedClass(proba, dBoundary = 0.5)
    return proba .>= dBoundary
end

#Check to see if yes or no
test_score = mean(Y_test .== predictedClass(predict_probability(X_test, θ)));
#results
println("Test score: ", round(test_score)
