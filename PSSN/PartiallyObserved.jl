#=
Probabilistic Spiking Neural Network Library
from https://arxiv.org/pdf/1910.01059.pdf

Default values are sensible for the Iris Dataset, expect to see about 80% accuracy for Iris and 90% for MNIST
=#

using Parameters        # Better struct printing using `@with_kw` macro
using Statistics        # Mean function
using ProgressLogging   # Training progress bar
using CUDA
using NNlib: softmax
using Random
using Printf


# Structs

"""
# A Network Layer
"""
@with_kw mutable struct Layer
    bias::Vector{Float64}
    r_weight::Vector{Float64}
    weights::Matrix{Float64}
end
"""
# A Network Layer implemented with CUDA Arrays
"""
@with_kw mutable struct cudaLayer
    bias::CuArray{Float32,1}
    r_weight::CuArray{Float32,1}
    weights::CuArray{Float32,2}
end
Layer(size, prev_size) = Layer(
    rand(size) * 2 .- 1,
    zeros(size),
    rand(Float64, (size, prev_size)) * 2 .- 1
)
Layer(layer::cudaLayer) = Layer(
    Array(layer.bias),
    Array(layer.r_weight),
    Array(layer.weights)
)
cudaLayer(layer::Layer) = cudaLayer(
    CuArray(layer.bias),
    CuArray(layer.r_weight),
    CuArray(layer.weights)
)

abstract type Network end

"""
# Fully Observed Network
"""
@with_kw mutable struct FONetwork <: Network
    output::Layer
    forward_k::Float64
    backward_k::Float64
end
"""
# Fully Observed Network
"""
Network(inputs::Int64, outputs::Int64, forward_k::Float64=0.625, backward_k::Float64=0.625) =
    FONetwork(
        Layer(outputs, inputs),
        forward_k,
        backward_k
    )

"""
# Partially Observed Network
"""
@with_kw mutable struct PONetwork <: Network
    layers::Vector{Layer}
    forward_k::Float64
    backward_k::Float64
end
"""
# Partially Observed Network implemented using CUDA Arrays
"""
@with_kw mutable struct cudaPONetwork <: Network
    layers::Vector{cudaLayer}
    forward_k::Float64
    backward_k::Float64
end
"""
# Partially Observed Network
"""
Network(inputs::Int64, outputs::Int64, hidden::Vector{Int64}, forward_k::Float64=0.625, backward_k::Float64=0.625) =
    PONetwork(
        vcat(Layer(hidden[1], inputs), [Layer(hidden[n], hidden[n-1]) for n in eachindex(hidden[2:end]) .+ 1], Layer(outputs, hidden[end])),
        forward_k,
        backward_k
    )
PONetwork(network::cudaPONetwork) = PONetwork(
    [Layer(layer) for layer in network.layers],
    network.forward_k,
    network.backward_k
)
cudaPONetwork(network::PONetwork) = cudaPONetwork(
    [cudaLayer(layer) for layer in network.layers],
    network.forward_k,
    network.backward_k
)


# Functions

## Fully Observed

"""
# Fully Observed

Evaluates a fully observed network for a given input

## Args

`network`: A fully observed network

`x[neuron, time, sample]`: Known input neuron values

## Return

`sigma[neuron, time, sample], spikes[neuron, time, sample]`: The firing probability and generated spikes of each neuron over the evaluation period

## Method
"""
function evaluate(network::FONetwork, x::AbstractArray{Bool,3})

    N, T, S = size(x)
    sigma = zeros(Float64, length(network.output.bias), T, S)
    spikes = zeros(Bool, length(network.output.bias), T, S)
    f_trace = zeros(N, S)
    b_trace = zeros(length(network.output.r_weight), S)

    for t = 2:T
        f_trace = trace_step(f_trace, (@view x[:, t-1, :]), network.forward_k)
        b_trace = trace_step(b_trace, (@view spikes[:, t-1, :]), network.backward_k)
        sigma[:, t, :] = σ(network.output, f_trace, b_trace)
        spikes[:, t, :] = spike(@view sigma[:, t, :])
    end

    sigma, spikes
end


"""
# Fully Observed

Batch trains a fully observed network

## Args

`network`: a fully observed network

`train_x[neuron, time, sample]`: Training known input neuron values

`train_y[neuron, time, sample]`: Training known output neuron values

`test_x[neuron, time, sample]`: Test known input neuron values

`test_y[neuron, time, sample]`: Test known output neuron values

`lr`: learning rate

`epochs`: number of training loops to complete

`minibatch_size`: Size of minibatch to use. Default value of `0` uses the whole batch

## Return

`accuracy[epoch], loss[epoch]`: Arrays with accuracy and loss values for each epoch

## Method
"""
function train!(network::FONetwork, train_x::AbstractArray{Bool,3}, train_y::AbstractArray{Bool,3}, test_x::AbstractArray{Bool,3}, test_y::AbstractArray{Bool,3}; minibatch_size::Int64=0, lr::Float64=0.01, epochs::Int64=80)

    @printf "Training: Fully Observed\n"
    accuracy_history = []
    loss_history = []
    mb = minibatch(train_x, train_y, minibatch_size)

    @progress for epoch in 1:epochs

        # Accuracy and loss
        sigma, spikes = evaluate(network, test_x)
        accuracy = mean(decode(spikes) .== decode(test_y)) * 100
        loss = log_p(test_y, sigma)
        append!(accuracy_history, accuracy)
        append!(loss_history, loss)
        @printf "%9i a:%.2f l:%.2f\n" epoch accuracy loss

        @progress for (x, y) in mb

            # Reset training variables
            N, T, S = size(x)
            f_trace = zeros(N, S)
            b_trace = zeros(length(network.output.r_weight), S)
            ∇bias = zeros(length(network.output.bias), S)
            ∇weights = zeros(size(network.output.weights))
            ∇r_weight = zeros(length(network.output.r_weight), S)

            # Compute gradients
            for t = 2:T
                f_trace = trace_step(f_trace, (@view x[:, t-1, :]), network.forward_k)
                b_trace = trace_step(b_trace, (@view y[:, t-1, :]), network.backward_k)
                post = y[:, t, :] - σ(network.output, f_trace, b_trace)
                ∇bias += post
                ∇r_weight += post .* b_trace
                ∇weights += post * f_trace'
            end

            # Update parameters
            network.output.bias += (lr / (T - 1)) .* vec(mean(∇bias, dims=2))
            network.output.r_weight += (lr / (T - 1)) .* vec(mean(∇r_weight, dims=2))
            network.output.weights += (lr / S / (T - 1)) .* ∇weights

            network.output.r_weight[network.output.r_weight.>0] .= 0
        end
    end

    accuracy_history, loss_history
end

"""
# Fully Observed

Batch trains a fully observed network using the momentum method

See Algorithm 4 from https://arxiv.org/pdf/2103.01327.pdf

## Args

`network`: Fully observed network

`train_x[neuron, time, sample]`: Training known input neuron values

`train_y[neuron, time, sample]`: Training known output neuron values

`test_x[neuron, time, sample]`: Test known input neuron values

`test_y[neuron, time, sample]`: Test known output neuron values

`lr`: Initial learning rate

`τ`: Initial learning period

`patience`: Number of training loops to complete after reaching max LB

`β1`: Gradient time averaging constant

`β2`: Gradient squared time averaging constant

`minibatch_size`: Size of minibatch to use. Default value of `0` uses the whole batch

## Return

`accuracy[epoch], loss[epoch]`: Arrays with accuracy and loss values for each epoch

## Method
"""
function train_m!(network::FONetwork, train_x::AbstractArray{Bool,3}, train_y::AbstractArray{Bool,3}, test_x::AbstractArray{Bool,3}, test_y::AbstractArray{Bool,3}; minibatch_size=0, τ::Int64=50, lr::Float64=0.01, β1::Float64=0.0, β2::Float64=0.0, patience::Int64=50)

    @printf "Training: Fully Observed with momentum\n"
    ϵ = lr
    mb = minibatch(train_x, train_y, minibatch_size)
    accuracy_history = []
    loss_history = []
    P = patience
    max_LB = -Inf
    epoch = 0

    while P > 0

        # Update epoch
        epoch += 1

        # Adapt leaning rate
        if τ < epoch
            ϵ = lr * τ / epoch
        end

        # Accuracy and cost
        sigma, spikes = evaluate(network, test_x)
        accuracy = mean(decode(spikes) .== decode(test_y)) * 100
        loss = log_p(test_y, sigma)
        append!(accuracy_history, accuracy)
        append!(loss_history, loss)
        @printf "%9i p:%5i lr:%.6f a:%.2f l:%.2f max_l:%.2f\n" epoch P ϵ accuracy loss max_LB

        # Patience check
        if max_LB < loss
            P = patience
            max_LB = loss + 0.01 # Stops it getting stuck making very small good updates indefinately
        else
            P -= 1
        end

        @progress for (x, y) in mb

            # Reset training variables
            N, T, S = size(x)
            f_trace = zeros(N, S)
            b_trace = zeros(length(network.output.r_weight), S)

            ∇bias = zeros(length(network.output.bias), S)
            ∇weights = zeros(size(network.output.weights))
            ∇r_weight = zeros(length(network.output.r_weight), S)

            ∇bias_bar = zeros(length(network.output.bias))
            ∇weights_bar = zeros(size(network.output.weights))
            ∇r_weight_bar = zeros(length(network.output.r_weight))
            ∇bias²_bar = zeros(length(network.output.bias))
            ∇weights²_bar = zeros(size(network.output.weights))
            ∇r_weight²_bar = zeros(length(network.output.r_weight))


            # Compute gradients
            for t = 2:T
                f_trace = trace_step(f_trace, (@view x[:, t-1, :]), network.forward_k)
                b_trace = trace_step(b_trace, (@view y[:, t-1, :]), network.backward_k)
                post = y[:, t, :] - σ(network.output, f_trace, b_trace)
                ∇bias += post
                ∇r_weight += post .* b_trace
                ∇weights += post * f_trace'
            end

            # Caluclate g and v as g^2
            ∇bias = vec(mean(∇bias, dims=2)) ./ (T - 1) # Averaging over the samples
            ∇r_weight = vec(mean(∇r_weight, dims=2)) ./ (T - 1)
            ∇r_weight = ∇r_weight ./ (S * (T - 1))

            ∇bias² = ∇bias .^ 2
            ∇weights² = ∇weights .^ 2
            ∇r_weight² = ∇r_weight .^ 2

            # Update ḡ and v̄
            ∇bias_bar = β1 * ∇bias_bar + (1 - β1) * ∇bias
            ∇weights_bar = β1 * ∇weights_bar + (1 - β1) * ∇weights
            ∇r_weight_bar = β1 * ∇r_weight_bar + (1 - β1) * ∇r_weight

            ∇bias²_bar = β2 * ∇bias²_bar + (1 - β2) * ∇bias²
            ∇weights²_bar = β2 * ∇weights²_bar + (1 - β2) * ∇weights²
            ∇r_weight²_bar = β2 * ∇r_weight²_bar + (1 - β2) * ∇r_weight²

            # Update parameters
            network.output.bias += ϵ .* ∇bias_bar ./ .√∇bias²_bar
            network.output.r_weight += ϵ .* ∇r_weight_bar ./ .√∇r_weight²_bar
            network.output.weights += ϵ .* ∇weights_bar ./ .√∇weights²_bar

            network.output.r_weight[network.output.r_weight.>0] .= 0 # Removes non-negative Reccurent gradients, seems to train better this way
        end
    end

    accuracy_history, loss_history
end


## Partially Observed

"""
# Partially Observed

Evaluates a partially observed network with a given input

## Args

`network`: Partially observed network

`x[neuron, time, sample]`: Known input neuron values

## Return

`sigma[layer][neuron, time, sample], spikes[layer][neuron, time, sample]`: σ(u) and generated spikes for the whole network over the training set

## Method
"""
function evaluate(network::PONetwork, x::AbstractArray{Bool,3})

    N, T, S = size(x)
    spikes = [zeros(Bool, length(layer.bias), T, S) for layer in network.layers]
    sigma = [zeros(length(layer.bias), T, S) for layer in network.layers]
    f_trace = [zeros(size(layer.weights, 2), S) for layer in network.layers]
    b_trace = [zeros(length(layer.r_weight), S) for layer in network.layers]

    for t = 2:T
        f_trace[1] = trace_step(f_trace[1], (@view x[:, t-1, :]), network.forward_k)
        b_trace[1] = trace_step(b_trace[1], (@view spikes[1][:, t-1, :]), network.backward_k)
        sigma[1][:, t, :] = σ(network.layers[1], f_trace[1], b_trace[1])
        spikes[1][:, t, :] = spike(@view sigma[1][:, t, :])

        for i in eachindex(network.layers[2:end-1]) .+ 1
            f_trace[i] = trace_step(f_trace[i], (@view spikes[i-1][:, t-1, :]), network.forward_k)
            b_trace[i] = trace_step(b_trace[i], (@view spikes[i][:, t-1, :]), network.backward_k)
            sigma[i][:, t, :] = σ(network.layers[i], f_trace[i], b_trace[i])
            spikes[i][:, t, :] = spike(@view sigma[i][:, t, :])
        end

        f_trace[end] = trace_step(f_trace[end], (@view spikes[end-1][:, t-1, :]), network.forward_k)
        b_trace[end] = trace_step(b_trace[end], (@view spikes[end][:, t-1, :]), network.backward_k)
        sigma[end][:, t, :] = σ(network.layers[end], f_trace[end], b_trace[end])
        spikes[end][:, t, :] = spike(@view sigma[end][:, t, :])
    end

    sigma, spikes
end
function evaluate(network::cudaPONetwork, x::AbstractArray{Bool,3})

    N, T, S = size(x)
    spikes = [CUDA.zeros(Bool, length(layer.bias), T, S) for layer in network.layers]
    sigma = [CUDA.zeros(length(layer.bias), T, S) for layer in network.layers]
    f_trace = [CUDA.zeros(size(layer.weights, 2), S) for layer in network.layers]
    b_trace = [CUDA.zeros(length(layer.r_weight), S) for layer in network.layers]

    for t = 2:T
        f_trace[1] = trace_step(f_trace[1], (@view x[:, t-1, :]), network.forward_k)
        b_trace[1] = trace_step(b_trace[1], (@view spikes[1][:, t-1, :]), network.backward_k)
        sigma[1][:, t, :] = σ(network.layers[1], f_trace[1], b_trace[1])
        spikes[1][:, t, :] = spike(@view sigma[1][:, t, :])

        for i in eachindex(network.layers[2:end-1]) .+ 1
            f_trace[i] = trace_step(f_trace[i], (@view spikes[i-1][:, t-1, :]), network.forward_k)
            b_trace[i] = trace_step(b_trace[i], (@view spikes[i][:, t-1, :]), network.backward_k)
            sigma[i][:, t, :] = σ(network.layers[i], f_trace[i], b_trace[i])
            spikes[i][:, t, :] = spike(@view sigma[i][:, t, :])
        end

        f_trace[end] = trace_step(f_trace[end], (@view spikes[end-1][:, t-1, :]), network.forward_k)
        b_trace[end] = trace_step(b_trace[end], (@view spikes[end][:, t-1, :]), network.backward_k)
        sigma[end][:, t, :] = σ(network.layers[end], f_trace[end], b_trace[end])
        spikes[end][:, t, :] = spike(@view sigma[end][:, t, :])
    end

    sigma, spikes
end

"""
# Partially Observed

Batch trains a partially observed network

## Args

`network`: A fully observed network

`train_x[neuron, time, sample]`: Training known input neuron values

`train_y[neuron, time, sample]`: Training known output neuron values

`test_x[neuron, time, sample]`: Test known input neuron values

`test_y[neuron, time, sample]`: Test known output neuron values

`lr`: Learning rate

`k`: Variational learning rate

`epochs`: Number of training loops to complete

`minibatch_size`: Size of minibatch to use. Default value of `0` uses the whole batch

## Return

`accuracy[epoch], loss[epoch]`: Arrays with accuracy and loss values

## Method
"""
function train!(network::PONetwork, train_x::AbstractArray{Bool,3}, train_y::AbstractArray{Bool,3}, test_x::AbstractArray{Bool,3}, test_y::AbstractArray{Bool,3}; minibatch_size::Int64=0, lr::Float64=0.1, k::Float64=0.01, epochs::Int64=80)

    @printf "Training: Partially Observed\n"
    accuracy_history = []
    loss_history = []
    mb = minibatch(train_x, train_y, minibatch_size)

    @progress for epoch in 1:epochs

        # Accuracy and loss
        sigma, spikes = evaluate(network, test_x)
        accuracy = mean(decode(spikes[end]) .== decode(test_y)) * 100
        loss = log_p(test_y, sigma[end])
        append!(accuracy_history, accuracy)
        append!(loss_history, loss)
        @printf "%9i a:%.2f l:%.2f\n" epoch accuracy loss

        @progress for (x, y) in mb

            # Reset trainng variables
            N, T, S = size(x)
            f_trace = [zeros(size(layer.weights, 2), S) for layer in network.layers]
            b_trace = [zeros(length(layer.r_weight), S) for layer in network.layers]
            ∇bias = [zeros(length(layer.bias), S) for layer in network.layers]
            ∇weights = [zeros(size(layer.weights)) for layer in network.layers]
            ∇r_weight = [zeros(length(layer.r_weight), S) for layer in network.layers]

            # Generate spikes
            sigma, spikes = evaluate(network, x)

            # Learning signal
            l = log_p(y, sigma[end])

            # Compute gradients
            for t = 2:T
                f_trace[1] = trace_step(f_trace[1], (@view x[:, t-1, :]), network.forward_k)
                b_trace[1] = trace_step(b_trace[1], (@view spikes[1][:, t-1, :]), network.backward_k)
                post = spikes[1][:, t, :] - sigma[1][:, t, :]
                ∇bias[1] += post
                ∇weights[1] += post * f_trace[1]'
                ∇r_weight[1] += post .* b_trace[1]

                for j in eachindex(network.layers[2:end-1]) .+ 1
                    f_trace[j] = trace_step(f_trace[j], (@view spikes[j-1][:, t-1, :]), network.forward_k)
                    b_trace[i] = trace_step(b_trace[i], (@view spikes[i][:, t-1, :]), network.backward_k)
                    post = spikes[j][:, t, :] - sigma[j][:, t, :]
                    ∇bias[j] += post
                    ∇weights[j] += post * f_trace[j]'
                    ∇r_weight[j] += post .* b_trace[j]
                end

                f_trace[end] = trace_step(f_trace[end], (@view spikes[end-1][:, t-1, :]), network.forward_k)
                b_trace[end] = trace_step(b_trace[end], (@view spikes[end][:, t-1, :]), network.backward_k)
                post = y[:, t, :] - sigma[end][:, t, :]
                ∇bias[end] += post
                ∇weights[end] += post * f_trace[end]'
                ∇r_weight[end] += post .* b_trace[end]
            end

            # Update parameter
            for i in eachindex(network.layers[1:end-1])
                network.layers[i].bias += (k * lr / (T - 1) * l) .* vec(mean(∇bias[i], dims=2))
                network.layers[i].weights += (k * lr / S / (T - 1) * l) .* ∇weights[i]
                network.layers[i].r_weight += (k * lr / (T - 1) * l) .* vec(mean(∇r_weight[i], dims=2))

                network.layers[i].r_weight[network.layers[i].r_weight.>0] .= 0
            end

            network.layers[end].bias += (lr / (T - 1)) .* vec(mean(∇bias[end], dims=2))
            network.layers[end].weights += (lr / S / (T - 1)) .* ∇weights[end]
            network.layers[end].r_weight += (lr / (T - 1)) .* vec(mean(∇r_weight[end], dims=2))

            network.layers[end].r_weight[network.layers[end].r_weight.>0] .= 0
        end
    end

    accuracy_history, loss_history
end

"""
# Partially Observed

Batch trains a partially observed network using CUDA

## Args

`network`: Partially observed network

`input_data[batch, time, neuon]`: Known input neuron values

`output_data[batch, time, neuon]`: Kknown output neuron values

`lr`: Learning rate

`k`: Variational learning rate

`epochs`: Number of training loops to complete

## Return

`accuracy[epoch], loss[epoch]`: Arrays with accuracy and loss values determined at 100 epochs

## Method
"""
function train_cuda!(network_original::PONetwork, train_x::AbstractArray{Bool,3}, train_y::AbstractArray{Bool,3}, test_x::AbstractArray{Bool,3}, test_y::AbstractArray{Bool,3}; minibatch_size::Int64=0, lr::Float64=0.1, k::Float64=0.01, epochs::Int64=80)

    @printf "Training: Partially Observed with CUDA\n"
    accuracy_history = []
    loss_history = []
    mb = minibatch(train_x, train_y, minibatch_size)
    network = cudaPONetwork(network_original)
    test_x = CuArray(test_x)
    test_y = CuArray(test_y)

    @progress for epoch in 1:epochs

        # Accuracy and loss
        sigma, spikes = evaluate(network, test_x)
        accuracy = mean(decode(spikes[end]) .== decode(test_y)) * 100
        loss = log_p(test_y, sigma[end])
        append!(accuracy_history, accuracy)
        append!(loss_history, loss)
        @printf "%9i a:%.2f l:%.2f\n" epoch accuracy loss

        @progress for (x, y) in mb

            x = CuArray(x)
            y = CuArray(y)

            # Reset trainng variables
            N, T, S = size(x)
            f_trace = [CUDA.zeros(size(layer.weights, 2), S) for layer in network.layers]
            b_trace = [CUDA.zeros(length(layer.r_weight), S) for layer in network.layers]
            ∇bias = [CUDA.zeros(length(layer.bias), S) for layer in network.layers]
            ∇weights = [CUDA.zeros(size(layer.weights)) for layer in network.layers]
            ∇r_weight = [CUDA.zeros(length(layer.r_weight), S) for layer in network.layers]

            # Generate spikes
            sigma, spikes = evaluate(network, x)

            # Learning signal
            l = log_p(y, sigma[end])

            # Compute gradients
            for t = 2:T
                f_trace[1] = trace_step(f_trace[1], (@view x[:, t-1, :]), network.forward_k)
                b_trace[1] = trace_step(b_trace[1], (@view spikes[1][:, t-1, :]), network.backward_k)
                post = spikes[1][:, t, :] - sigma[1][:, t, :]
                ∇bias[1] += post
                ∇weights[1] += post * f_trace[1]'
                ∇r_weight[1] += post .* b_trace[1]

                for j in eachindex(network.layers[2:end-1]) .+ 1
                    f_trace[j] = trace_step(f_trace[j], (@view spikes[j-1][:, t-1, :]), network.forward_k)
                    b_trace[i] = trace_step(b_trace[i], (@view spikes[i][:, t-1, :]), network.backward_k)
                    post = spikes[j][:, t, :] - sigma[j][:, t, :]
                    ∇bias[j] += post
                    ∇weights[j] += post * f_trace[j]'
                    ∇r_weight[j] += post .* b_trace[j]
                end

                f_trace[end] = trace_step(f_trace[end], (@view spikes[end-1][:, t-1, :]), network.forward_k)
                b_trace[end] = trace_step(b_trace[end], (@view spikes[end][:, t-1, :]), network.backward_k)
                post = y[:, t, :] - sigma[end][:, t, :]
                ∇bias[end] += post
                ∇weights[end] += post * f_trace[end]'
                ∇r_weight[end] += post .* b_trace[end]
            end

            # Update parameter
            for i in eachindex(network.layers[1:end-1])
                network.layers[i].bias += k * lr / (T - 1) * l .* vec(mean(∇bias[i], dims=2))
                network.layers[i].weights += k * lr / S / (T - 1) * l .* ∇weights[i]
                network.layers[i].r_weight += (k * lr / (T - 1) * l) .* vec(mean(∇r_weight[i], dims=2))

                network.layers[i].r_weight[network.layers[i].r_weight.>0] .= 0
            end

            network.layers[end].bias += lr / (T - 1) .* vec(mean(∇bias[end], dims=2))
            network.layers[end].weights += lr / S / (T - 1) .* ∇weights[end]
            network.layers[end].r_weight += (lr / (T - 1)) .* vec(mean(∇r_weight[end], dims=2))

            network.layers[end].r_weight[network.layers[end].r_weight.>0] .= 0
        end
    end

    network_original = PONetwork(network)

    accuracy_history, loss_history
end

"""
# Partially Observed

Batch trains a partially observed network with multiple samples using Expectation Maximisation
https://arxiv.org/pdf/2102.03280.pdf

## Args

`network`: Partially observed network

`input_data[batch, time, neuon]`: Known input neuron values

`output_data[batch, time, neuon]`: Kknown output neuron values

`lr`: Learning rate

`k`: Variational learning rate

`epochs`: Number of training loops to complete

`S`: Number of samples to use

## Return

`accuracy[epoch], loss[epoch]`: Arrays with accuracy and loss values determined at 100 epochs

## Method
"""
function train_em!(network::PONetwork, train_x::AbstractArray{Bool,3}, train_y::AbstractArray{Bool,3}, test_x::AbstractArray{Bool,3}, test_y::AbstractArray{Bool,3}; minibatch_size::Int64=0, C::Int64=5, lr::Float64=0.1, epochs::Int64=80)

    print("Training: Partially Observed with Expectation Maximisation using $C Compartments\n")
    accuracy_history = []
    loss_history = []
    l = zeros(C)
    mb = minibatch(train_x, train_y, minibatch_size)

    @progress for epoch in 0:epochs-1

        # Accuracy and loss
        sigma, spikes = evaluate(network, test_x)
        accuracy = mean(decode(spikes[end]) .== decode(test_y)) * 100
        loss = log_p(test_y, sigma[end])
        append!(accuracy_history, accuracy)
        append!(loss_history, loss)
        @printf "%9i a:%.2f l:%.2f\n" epoch accuracy loss

        @progress for (x, y) in mb

            # Reset trainng variables
            N, T, S = size(x)
            ∇bias = [zeros(length(layer.bias), S) for c = 1:C, layer in network.layers]
            ∇weights = [zeros(size(layer.weights)...) for c = 1:C, layer in network.layers]
            ∇r_weight = [zeros(length(layer.r_weight), S) for c = 1:C, layer in network.layers]

            # Compartments
            for c = 1:C

                f_trace = [zeros(size(layer.weights, 2), S) for layer in network.layers]
                b_trace = [zeros(length(layer.r_weight), S) for layer in network.layers]

                # Generate spikes
                sigma, spikes = evaluate(network, x)

                # Learning signal
                l[c] = log_p(y, sigma[end])

                # Compute gradients
                for t = 2:T
                    f_trace[1] = trace_step(f_trace[1], (@view x[:, t-1, :]), network.forward_k)
                    b_trace[1] = trace_step(b_trace[1], (@view spikes[1][:, t-1, :]), network.backward_k)
                    post = spikes[1][:, t, :] - sigma[1][:, t, :]
                    ∇bias[c, 1] += post
                    ∇weights[c, 1] += post * f_trace[1]'
                    ∇r_weight[c, 1] += post .* b_trace[1]

                    for j in eachindex(network.layers[2:end-1]) .+ 1
                        f_trace[j] = trace_step(f_trace[j], (@view spikes[j-1][:, t-1, :]), network.forward_k)
                        b_trace[i] = trace_step(b_trace[i], (@view spikes[i][:, t-1, :]), network.backward_k)
                        post = spikes[j][:, t, :] - sigma[j][:, t, :]
                        ∇bias[c, j] += post
                        ∇weights[c, j] += post * f_trace[j]'
                        ∇r_weight[c, j] += post .* b_trace[j]
                    end

                    f_trace[end] = trace_step(f_trace[end], (@view spikes[end-1][:, t-1, :]), network.forward_k)
                    b_trace[end] = trace_step(b_trace[end], (@view spikes[end][:, t-1, :]), network.backward_k)
                    post = y[:, t, :] - sigma[end][:, t, :]
                    ∇bias[c, end] += post
                    ∇weights[c, end] += post * f_trace[end]'
                    ∇r_weight[c, end] += post .* b_trace[end]
                end
            end

            # Softmax learning signal
            l = softmax(l)

            # Update parameter
            for i in eachindex(network.layers)
                network.layers[i].bias += (lr / (T - 1)) .* sum([l[c] .* vec(mean(∇bias[c, i], dims=2)) for c = 1:C])
                network.layers[i].weights += (lr / S / (T - 1)) .* sum([l[c] .* ∇weights[c, i] for c = 1:C])
                network.layers[i].r_weight += (lr / (T - 1)) .* sum([l[c] .* vec(mean(∇r_weight[c, i], dims=2)) for c = 1:C])

                network.layers[i].r_weight[network.layers[i].r_weight.>0] .= 0
            end
        end
    end

    accuracy_history, loss_history
end

function train_em_m!(network::PONetwork, train_x::AbstractArray{Bool,3}, train_y::AbstractArray{Bool,3}, test_x::AbstractArray{Bool,3}, test_y::AbstractArray{Bool,3}; τ::Int64=10, patience::Int64=20, minibatch_size::Int64=0, C::Int64=5, β1::Float64=0., β2::Float64=0., lr::Float64=0.1)

    print("Training Started: Partially Observed with Expectation Maximisation using $C Compartments - β1:$β1 β2:$β2 lr:$lr τ:$τ patience:$patience minibatch_size=$minibatch_size\n")
    ϵ = lr
    accuracy_history = []
    loss_history = []
    l = zeros(C)
    mb = minibatch(train_x, train_y, minibatch_size)
    epoch = 0
    P = patience
    max_LB = -Inf

    while P > 0

        # Update epoch
        epoch += 1

        # Adapt leaning rate
        # if τ < epoch
        #     ϵ = lr * τ / epoch
        # end
        ϵ = ϵ * 0.97

        # Accuracy and cost
        sigma, spikes = evaluate(network, test_x)
        accuracy = mean(decode(spikes[end]) .== decode(test_y)) * 100
        loss = log_p(test_y, sigma[end])
        append!(accuracy_history, accuracy)
        append!(loss_history, loss)
        @printf "%9i p:%5i lr:%.6f a:%.2f l:%.2f max_l:%.2f\n" epoch P ϵ accuracy loss max_LB

        # Patience check
        if max_LB < loss
            P = patience
            max_LB = loss + 0.01 # Stops it getting stuck making very small good updates indefinately
        else
            P -= 1
        end

        @progress for (x, y) in mb

            # Reset trainng variables
            N, T, S = size(x)
            ∇bias = [zeros(length(layer.bias), S) for c = 1:C, layer in network.layers]
            ∇weights = [zeros(size(layer.weights)) for c = 1:C, layer in network.layers]
            ∇r_weight = [zeros(length(layer.r_weight), S) for c = 1:C, layer in network.layers]

            ∇bias_bar = [zeros(length(layer.bias)) for layer in network.layers]
            ∇weights_bar = [zeros(size(layer.weights)) for layer in network.layers]
            ∇r_weight_bar = [zeros(length(layer.r_weight)) for layer in network.layers]
            ∇bias²_bar = [zeros(length(layer.bias)) for layer in network.layers]
            ∇weights²_bar = [zeros(size(layer.weights)) for layer in network.layers]
            ∇r_weight²_bar = [zeros(length(layer.r_weight)) for layer in network.layers]

            # Compartments
            for c = 1:C

                f_trace = [zeros(size(layer.weights, 2), S) for layer in network.layers]
                b_trace = [zeros(length(layer.r_weight), S) for layer in network.layers]

                # Generate spikes
                sigma, spikes = evaluate(network, x)

                # Learning signal
                l[c] = log_p(y, sigma[end])

                # Compute gradients
                for t = 2:T
                    f_trace[1] = trace_step(f_trace[1], (@view x[:, t-1, :]), network.forward_k)
                    b_trace[1] = trace_step(b_trace[1], (@view spikes[1][:, t-1, :]), network.backward_k)
                    post = spikes[1][:, t, :] - sigma[1][:, t, :]
                    ∇bias[c, 1] += post
                    ∇weights[c, 1] += post * f_trace[1]'
                    ∇r_weight[c, 1] += post .* b_trace[1]

                    for j in eachindex(network.layers[2:end-1]) .+ 1
                        f_trace[j] = trace_step(f_trace[j], (@view spikes[j-1][:, t-1, :]), network.forward_k)
                        b_trace[i] = trace_step(b_trace[i], (@view spikes[i][:, t-1, :]), network.backward_k)
                        post = spikes[j][:, t, :] - sigma[j][:, t, :]
                        ∇bias[c, j] += post
                        ∇weights[c, j] += post * f_trace[j]'
                        ∇r_weight[c, j] += post .* b_trace[j]
                    end

                    f_trace[end] = trace_step(f_trace[end], (@view spikes[end-1][:, t-1, :]), network.forward_k)
                    b_trace[end] = trace_step(b_trace[end], (@view spikes[end][:, t-1, :]), network.backward_k)
                    post = y[:, t, :] - sigma[end][:, t, :]
                    ∇bias[c, end] += post
                    ∇weights[c, end] += post * f_trace[end]'
                    ∇r_weight[c, end] += post .* b_trace[end]
                end
            end

            # Softmax learning signal
            l = softmax(l)

            # Sum over softmax to find most likely Gradient
            ∇bias = [sum([l[c] .* vec(mean(∇bias[c, i], dims=2)) for c = 1:C]) ./ (T - 1) for i in eachindex(network.layers)]# Average gradients over Compartments and Samples
            ∇weights = [sum([l[c] .* ∇weights[c, i] for c = 1:C]) ./ (S * (T - 1)) for i in eachindex(network.layers)]
            ∇r_weight = [sum([l[c] .* vec(mean(∇r_weight[c, i], dims=2)) for c = 1:C]) ./ (T - 1) for i in eachindex(network.layers)]

            for i in eachindex(network.layers)

                # Caluclate v as g^2
                ∇bias² = ∇bias[i] .^ 2
                ∇weights² = ∇weights[i] .^ 2
                ∇r_weight² = ∇r_weight[i] .^ 2

                # Update ḡ and v̄
                ∇bias_bar[i] = β1 .* ∇bias_bar[i] .+ (1 - β1) .* ∇bias[i]
                ∇weights_bar[i] = β1 .* ∇weights_bar[i] .+ (1 - β1) .* ∇weights[i]
                ∇r_weight_bar[i] = β1 .* ∇r_weight_bar[i] .+ (1 - β1) .* ∇r_weight[i]

                ∇bias²_bar[i] = β2 .* ∇bias²_bar[i] .+ (1 - β2) .* ∇bias²
                ∇weights²_bar[i] = β2 .* ∇weights²_bar[i] .+ (1 - β2) .* ∇weights²
                ∇r_weight²_bar[i] = β2 .* ∇r_weight²_bar[i] .+ (1 - β2) .* ∇r_weight²

                # Update parameter
                network.layers[i].bias += ϵ .* ∇bias_bar[i] ./ .√∇bias²_bar[i]
                network.layers[i].weights += ϵ .* ∇weights_bar[i] ./ .√∇weights²_bar[i]
                network.layers[i].r_weight += ϵ .* ∇r_weight_bar[i] ./ .√∇r_weight²_bar[i]

                network.layers[i].r_weight[network.layers[i].r_weight.>0] .= 0
            end
        end
    end

    print("Training Complete: Partially Observed with Expectation Maximisation using $C Compartments - β1:$β1 β2:$β2 lr:$lr τ:$τ patience:$patience minibatch_size=$minibatch_size\n")

    accuracy_history, loss_history
end


## General Functions

"""
Updates the value of a kernel trace
"""
function trace_step(trace::AbstractMatrix{Float64}, spikes::AbstractMatrix{Bool}, k::Float64)
    k * trace + spikes
end
function trace_step(trace::CuArray, spikes::SubArray, k::Float64)
    k * trace + spikes
end

"""
Generates spikes from a firing probability
"""
function spike(x)
    x .> rand()
end

"""
Calculate the firing probability
"""
function σ(layer::Layer, f_trace::AbstractArray{Float64}, b_trace::AbstractArray{Float64})
    σ.(layer.weights * f_trace .+ layer.r_weight .* b_trace .+ layer.bias)
end
function σ(layer::cudaLayer, f_trace::CuArray)
    σ.(layer.weights * f_trace .+ layer.r_weight .* b_trace .+ layer.bias)
end

"""
# Sigmoid function
"""
function σ(x)
    1 / (1 + exp(-x))
end

# log_px(x, u) = -log((1 + exp(-sign(x - 1 / 2) * u)))

function log_p(x, σ)
    s = σ[:, 2:end, :]
    x = x[:, 2:end, :]
    s[x.==0] = 1 .- s[x.==0]
    s = log.(s)
    s[s.==-Inf] .= -9999
    mean(sum(s, dims=(1, 2)))
end

"""
Inplace convolution, not currently used
"""
function iconv(s, k)
    l = length(s)
    sum([l >= n ? k[n] * s[end-n+1] : 0 for n = eachindex(k)])
end


## Data Wrangling

label_encode(y, T) = [b == c for c = sort(unique(y)), t = 1:T, b = y]

function poisson_encode(input, T, max_rate=0.5)
    out = rand(Float64, (size(input)..., T)) .< max_rate * input
    permutedims(Array(out), (1, 3, 2))
end

function logsoftmax(x)
    x .-= maximum(x)
end

decode(x) = mapslices(argmax, sum(x, dims=2), dims=1)

function spike_accuracy(y, ŷ)
    mean(y .== ŷ)
end

"""
Normalises data with μ 0 and σ 1
"""
function normalize_iris(input)
    for (i, feature) in enumerate(eachcol(input))
        x̄ = mean(feature)
        sd = std(feature)
        x = (feature .- x̄) / sd
        input[:, i] = σ.(x)
    end
    input
end

"""
Generates a minibatch from a batch

Useful for large datasets
"""
minibatch(input_data::AbstractArray{Bool,3}, output_data::AbstractArray{Bool,3}, minibatch_size::Int64) =
    [(input_data[:, :, i], output_data[:, :, i]) for i in Iterators.partition(Random.shuffle(collect(1:size(input_data, 3))), (minibatch_size > 0 ? minibatch_size : size(input_data, 3)))]

