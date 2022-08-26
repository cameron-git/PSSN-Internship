include("PSSN.jl") # Library include

using Plots
using FileIO           # Saving and Loading premade training data, see DataEncoding.jl

# Constants
INPUTS = 4
HIDDEN = [50]
OUTPUTS = 3
lr = 0.1
k = 0.001
epochs = 200
# file = "../data/mnist/mnist.jld2"
file = "../data/iris/iris.jld2"

# Data Loading
train_x = load(file, "train_x")#[:,1:10,:]
train_y = load(file, "train_y")#[:,1:10,:]
test_x = load(file, "test_x")#[:,1:10,:]
test_y = load(file, "test_y")#[:,1:10,:]

# Defining the model
network = Network(INPUTS, OUTPUTS, HIDDEN)

# Training
accuracy_history, loss_history = @time train_em_m!(network, train_x, train_y, test_x, test_y; C=3, lr=lr)

# Plot training accuracy and loss progress
display(plot(ylims=(-250, 0), loss_history, title="lr=$lr k=$k", label=nothing, xlabel="Epoch", ylabel="Loss"))
display(plot(ylims=(0, 100), accuracy_history, title="lr=$lr k=$k", label=nothing, xlabel="Epoch", ylabel="Accuracy %"))

# Average spike accuracy
let a = 0
    for i in 1:10
        _, spikes = evaluate(network, test_x)
        a += spike_accuracy(spikes[end], test_y)
    end
    a / 10
end

display(mean(accuracy_history[end-100:end]))

for layer in network.layers
    display(maximum(abs.(layer.bias)))
    display(mean(abs.(layer.bias)))
    display(maximum(abs.(layer.weights)))
    display(mean(abs.(layer.weights)))
    display(maximum(abs.(layer.r_weight)))
    display(mean(abs.(layer.r_weight)))
end

display(maximum(abs.(network.output.bias)))
display(mean(abs.(network.output.bias)))
display(maximum(abs.(network.output.weights)))
display(mean(abs.(network.output.weights)))
display(maximum(abs.(network.output.r_weight)))
display(mean(abs.(network.output.r_weight)))