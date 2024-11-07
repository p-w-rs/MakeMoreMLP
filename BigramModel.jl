using Lux, ComponentArrays, Optimisers, Zygote, Random, OneHotArrays, MLUtils
using StatsBase, Plots

rng = MersenneTwister()
Random.seed!(rng, 12345)

# Get the Data
words = open("assets/names.txt") do file
    readlines(file)
end

# Initial Data Analysis
counts = Dict{Tuple{String, String}, Int}()
for w in words
    aug_w = ["<S>", split(w, "")..., "<E>"]
    for bigram in zip(aug_w, aug_w[2:end])
        counts[bigram] = get(counts, bigram, 0) + 1
    end
end
sort(collect(counts), by=x->x[2], rev=true)

# Clean the Data for the Model
chars = vcat(map(w -> split(w, ""), words)...)
chars = [".", sort(unique(chars))...]
stoi = Dict([c => i for (i, c) in enumerate(chars)])
itos = Dict([i => c for (i, c) in enumerate(chars)])

# Build the Model
# We could start every value at a higher base count than zero to smooth the probabilities
cnts = zeros(Int, 27, 27)
for w in words
    aug_w = [".", split(w, "")..., "."]
    for (c1, c2) in zip(aug_w, aug_w[2:end])
        cnts[stoi[c1], stoi[c2]] += 1
    end
end
probs = cnts ./ sum(cnts, dims=2)

# Visualize the Model
heatmap(cnts,
    xticks=(1:27, [string(itos[i]) for i in 1:27]),
    yticks=(1:27, [string(itos[i]) for i in 1:27]),
    yflip=true, c=:blues
)

# Generate 20 Names (Bigram models are not very good at this task)
more = String[]
for i in 1:20
    name, c = [], 1
    while true
        c = sample(rng, 1:27, weights(probs[c, :]))
        c == 1 && break
        push!(name, itos[c])
    end
    push!(more, join(name))
end
println("Direct Porbabalistic Model:\n", more, "\n\n")

# How do I know it is better than random? Check it!
more = String[]
for i in 1:20
    name, c = [], 1
    while true
        c = rand(rng, 1:27)
        c == 1 && break
        push!(name, itos[c])
    end
    push!(more, join(name))
end
println("Random Model:\n", more, "\n\n")

# Now lets train an MLP model to generate names
# First we need to convert the data to a format that the model can understand
xs, ys = Int[], Int[]
for w in words
    aug_w = [".", split(w, "")..., "."]
    for (c1, c2) in zip(aug_w, aug_w[2:end])
        push!(xs, stoi[c1])
        push!(ys, stoi[c2])
    end
end
xs = Float32.(onehotbatch(xs, 1:27))
ys = Float32.(onehotbatch(ys, 1:27))
loader = DataLoader((xs, ys), batchsize=32, shuffle=true)

# Now lets define our model, we will use one weight matrix in hopes of matching our bigram model we calculated above
model = Dense(27 => 27)
ps, st = Lux.setup(rng, model)
ps = ComponentArray(ps)
opt = Optimisers.setup(Optimisers.Adam(0.0001f0), ps)

for epoch in 1:150
    global model, ps, st, opt
    losses = Float32[]
    for (x, y) in loader
        loss = 0.0
        grads = Zygote.gradient(ps, st) do ps, st
            logits, st = model(x, ps, st)

            # Softmax
            local counts = exp.(logits)
            local probs = counts ./ sum(counts, dims=1)

            # Negative Log Likelihood
            loss = -sum(y .* log.(probs)) / size(x, 2)

            # Smooth probabilities via regularization on ps
            # This is the same as smoothing the probabilities in the bigram model
            rg_loss = loss + (0.001f0 * sum(ps.^2))
        end
        opt, ps = Optimisers.update(opt, ps, grads[1])
        push!(losses, loss)
    end
    println("Epoch: $epoch, Loss: $(mean(losses))")
end

# Generate 20 Names (Bigram models are not very good at this task)
more = String[]
for i in 1:20
    name, x = [], Float32.(onehot(".", chars))
    while true
        logits, _ = model(x, ps, st)
        local counts = exp.(logits)
        local probs = counts ./ sum(counts, dims=1)
        c = sample(rng, 1:27, weights(probs))
        c == 1 && break
        push!(name, itos[c])
        x = Float32.(onehot(itos[c], chars))
    end
    push!(more, join(name))
end
println("MLP Model:\n", more, "\n\n")
