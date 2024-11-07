using Lux, ComponentArrays, Optimisers, Zygote, Random, OneHotArrays, MLUtils
using DataStructures, StatsBase, Plots

rng = Xoshiro()
Random.seed!(rng, 12345)
VOCAB_SIZE = 56
BLOCK_SIZE = 128
CONV_SIZE = 2
BATCH_SIZE = 128

# Get the Data
bible = open("assets/bible.txt") do file
    content = read(file, String)
    lowercase.(filter(isascii, collect(content)))
end

# Clean the Data for the Model
chars = vcat('\0', sort(unique(bible)))
stoi = Dict([c => i for (i, c) in enumerate(chars)])
itos = Dict([i => c for (i, c) in enumerate(chars)])

xs, ys = [], Int[]
context = CircularBuffer{Int}(BLOCK_SIZE)
fill!(context, 1)
for c in bible
    i = stoi[c]
    push!(xs, context[:])
    push!(ys, i)
    push!(context, i)
end
xs = Float32.(onehotbatch(hcat(xs...), 1:VOCAB_SIZE))
xs = reshape(xs, VOCAB_SIZE, BLOCK_SIZE, 1, :)
ys = Float32.(onehotbatch(ys, 1:VOCAB_SIZE))
loader = DataLoader((xs, ys), batchsize=BATCH_SIZE, shuffle=true)

# Build look-up tables / embeddings and Model
model = Chain(
    Conv((VOCAB_SIZE, CONV_SIZE), 1 => 32, relu; stride=2),
    Conv((1, CONV_SIZE), 32 => 64, relu; stride=2),
    Conv((1, CONV_SIZE), 64 => 128, relu; stride=2),
    Conv((1, CONV_SIZE), 128 => 256, relu; stride=2),
    FlattenLayer(),
    Dense(8 * 256 => VOCAB_SIZE)
)
ps, st = Lux.setup(rng, model)
ps = ComponentArray(ps)
opt = Optimisers.setup(Optimisers.Adam(0.01f0), ps)
logitcrossentropy(ŷ, y; dims=1, agg=mean) = agg(.-sum(y .* logsoftmax(ŷ; dims=dims); dims=dims))

# Train the Model
for epoch in 1:100
    global model, ps, st, opt
    global losses = Float32[]
    global loss = 0.0
    for (i, (x, y)) in enumerate(loader)
        grads = Zygote.gradient(ps, st) do ps, st
            logits, st = model(x, ps, st)
            loss = logitcrossentropy(logits, y)
        end
        opt, ps = Optimisers.update(opt, ps, grads[1])
        push!(losses, loss)
        i >= 100 && break
    end
    println("Epoch: $epoch, Loss: $(mean(losses))")
end

# Generate 1000 chars
more, x = Vector{Char}(undef, 1000), ones(Int32, BLOCK_SIZE, 1)
x = Float32.(onehotbatch(x, 1:VOCAB_SIZE))
x = reshape(x, VOCAB_SIZE, BLOCK_SIZE, 1, 1)
for i in 1:1000
    logits, st = model(x, ps, st)
    probs = softmax(logits)

    c = sample(rng, 1:VOCAB_SIZE, weights(probs[:]))
    more[i] = itos[c]

    c = Float32.(onehot(c, 1:VOCAB_SIZE))
    x = cat(x[:, 2:end, :, :], c, dims=2)
end
println("Generated Text: ", join(more))
