using MLDatasets
using DataFrames
using Statistics
using Random
using Images
using ImageInTerminal
using FileIO

include("PSSN.jl") # Library include

T = 32


# MNIST

train_mnist = MNIST(dir="../data/mnist")
test_mnist = MNIST(split=:test, dir="../data/mnist")

train_x = train_mnist.features#[:,:, mnist_train.targets .< 2]
train_x = permutedims(train_x, (2, 1, 3))
train_x = reshape(train_x, :, size(train_x, 3))

test_x = test_mnist.features#[:,:, mnist_train.targets .< 2]
test_x = permutedims(test_x, (2, 1, 3))
test_x = reshape(test_x, :, size(test_x, 3))


train_labels = train_mnist.targets#[mnist_train.targets .< 2]
test_labels = test_mnist.targets#[mnist_train.targets .< 2]


# Encoding

train_x = poisson_encode(train_x, T, 1.0)
test_x = poisson_encode(test_x, T, 1.0)

train_y = label_encode(train_labels, T)
test_y = label_encode(test_labels, T)

# i,j,k = size(train_y)
# train_y = [(rand() > 0.2) ? train_y[i,j,k] : !train_y[i,j,k] for i=1:i,j=1:j,k=1:k]
# i,j,k = size(test_y)
# test_y = [(rand() > 0.2) ? test_y[i,j,k] : !test_y[i,j,k] for i=1:i,j=1:j,k=1:k]

save(
    "../data/mnist/mnist.jld2",
    "train_x", train_x,
    "train_y", train_y,
    "train_labels", train_labels,
    "test_x", test_x,
    "test_y", test_y,
    "test_labels", test_labels
)



# Iris Dataset

T = 32

iris = Iris(; dir="../data/iris")
# vscodedisplay(iris.dataframe)

iris_x = Matrix(iris.features)

iris_x = normalize_iris(iris_x)'

iris_x = reduce(hcat, fill(iris_x, 100))

train_x = poisson_encode(iris_x, T, 1)
test_x = poisson_encode(iris_x, T, 1)

iris_labels = vec([(
    if i == "Iris-setosa"
        1
    elseif i == "Iris-versicolor"
        2
    elseif i == "Iris-virginica"
        3
    else
        0
    end
) for i = Matrix(iris.targets)])

iris_labels = reduce(vcat, fill(iris_labels,100))

train_y = label_encode(iris_labels, T)
test_y = label_encode(iris_labels, T)

save(
    "../data/iris/iris.jld2",
        "train_x", train_x,
        "train_y", train_y,
        "train_labels", iris_labels,
        "test_x", test_x,
        "test_y", test_y,
        "test_labels", iris_labels
)