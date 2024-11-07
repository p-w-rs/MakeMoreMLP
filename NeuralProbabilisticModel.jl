using Lux, ComponentArrays, Optimisers, Zygote, Random, OneHotArrays, MLUtils
using DataStructures, StatsBase, Plots

rng = MersenneTwister()
Random.seed!(rng, 12345)
VOCAB_SIZE = 27
BLOCK_SIZE = 3
EMB_SIZE = 10
H_SIZE = 256
BATCH_SIZE = 128

# Get the Data
words = open("assets/names.txt") do file
    readlines(file)
end

# Clean the Data for the Model
chars = vcat(map(w -> split(w, ""), words)...)
chars = [".", sort(unique(chars))...]
stoi = Dict([c => i for (i, c) in enumerate(chars)])
itos = Dict([i => c for (i, c) in enumerate(chars)])


xs, ys = [], Int[]
for w in words
    #println(w)
    context = CircularBuffer{Int}(BLOCK_SIZE)
    fill!(context, 1)
    for c in [split(w, "")..., "."]
        i = stoi[c]
        push!(xs, context[:])
        push!(ys, i)

        #println(join([itos[j] for j in context[:]]), " --> ", itos[i])
        push!(context, i)
    end
    #println()
end
xs = Int32.(hcat(xs...))
ys = Float32.(onehotbatch(ys, 1:length(chars)))
loader = DataLoader((xs, ys), batchsize=BATCH_SIZE, shuffle=true)

# Build look-up tables / embeddings and Model
model = Chain(
    Embedding(VOCAB_SIZE => EMB_SIZE),
    FlattenLayer(),
    Dense(EMB_SIZE*BLOCK_SIZE => H_SIZE),
    BatchNorm(H_SIZE, tanh),
    Dense(H_SIZE => length(chars))
)
ps, st = Lux.setup(rng, model)
ps = ComponentArray(ps)
st = Lux.trainmode(st)
opt = Optimisers.setup(Optimisers.Adam(0.0001f0), ps)

# Train the Model
for epoch in 1:1000
    global model, ps, st, opt
    global losses = Float32[]
    global loss = 0.0
    for (x, y) in loader
        grads = Zygote.gradient(ps, st) do ps, st
            logits, st = model(x, ps, st)

            # Softmax
            log_counts = exp.(logits)
            probs = log_counts ./ sum(log_counts, dims=1)

            # Negative Log Likelihood
            loss = -sum(y .* log.(probs)) / BATCH_SIZE

            # Smooth probabilities via regularization on ps
            # This is the same as smoothing the probabilities in the bigram model
            # rg_loss = loss + (0.001f0 * sum(ps.^2)) + (0.0001f0 * sum(ps))
        end
        opt, ps = Optimisers.update(opt, ps, grads[1])
        push!(losses, loss)
    end
    println("Epoch: $epoch, Loss: $(mean(losses))")
end

# Generate 20 Names
more = String[]
st = Lux.testmode(st)
for i in 1:20
    name, x = [], ones(Int32, BLOCK_SIZE, 1)
    while true
        logits, _ = model(x, ps, st)
        log_counts = exp.(logits)
        probs = log_counts ./ sum(log_counts, dims=1)
        c = sample(rng, 1:27, weights(probs[:]))
        c == 1 && break
        push!(name, itos[c])
        x = vcat(x[2:end, :], c)
    end
    push!(more, join(name))
end
println("Generated Names: ", more)
