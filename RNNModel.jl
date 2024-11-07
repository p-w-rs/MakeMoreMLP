using Lux, ComponentArrays, Optimisers, Zygote, Random, OneHotArrays, MLUtils
using DataStructures, StatsBase, Plots

rng = MersenneTwister()
Random.seed!(rng, 12345)
VOCAB_SIZE = 27
BLOCK_SIZE = 5
H_SIZE = 64
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
    context = CircularBuffer{Int}(BLOCK_SIZE)
    fill!(context, 1)
    for c in [split(w, "")..., "."]
        i = stoi[c]
        push!(xs, context[:])
        push!(ys, i)
        push!(context, i)
    end
end
xs = Float32.(onehotbatch(hcat(xs...), 1:VOCAB_SIZE))
ys = Float32.(onehotbatch(ys, 1:VOCAB_SIZE))
loader = DataLoader((xs, ys), batchsize=BATCH_SIZE, shuffle=true)

# Build look-up tables / embeddings and Model
model = Chain(
    Recurrence(RNNCell(VOCAB_SIZE => H_SIZE, sigmoid; train_state=true)),
    Dense(H_SIZE => VOCAB_SIZE)
)
ps, st = Lux.setup(rng, model)
ps = ComponentArray(ps)
opt = Optimisers.setup(Optimisers.Adam(0.01f0), ps)
logitcrossentropy(ŷ, y; dims = 1, agg = mean) = agg(.-sum(y .* logsoftmax(ŷ; dims = dims); dims = dims))

# Train the Model
for epoch in 1:1000
    global model, ps, st, opt
    global losses = Float32[]
    global loss = 0.0
    for (x, y) in loader
        grads = Zygote.gradient(ps, st) do ps, st
            logits, st = model(x, ps, st)
            loss = logitcrossentropy(logits, y)
        end
        opt, ps = Optimisers.update(opt, ps, grads[1])
        st = Lux.initialstates(rng, model)
        push!(losses, loss)
    end
    println("Epoch: $epoch, Loss: $(mean(losses))")
end

# Generate 20 Names
more = String[]
for i in 1:20
    name, x = [], ones(Int32, BLOCK_SIZE, 1)
    x = Float32.(onehotbatch(x, 1:VOCAB_SIZE))
    while true
        logits, st = model(x, ps, st)
        probs = softmax(logits)

        c = sample(rng, 1:27, weights(probs[:]))
        c == 1 && break
        push!(name, itos[c])

        c = Float32.(onehot(c, 1:VOCAB_SIZE))
        x = cat(x[:, 2:end, :], c, dims=2)
        st = Lux.initialstates(rng, model)
    end
    push!(more, join(name))
end
println("Generated Names: ", more)
