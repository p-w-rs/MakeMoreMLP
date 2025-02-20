using Lux, ComponentArrays, Optimisers, Zygote, Random, OneHotArrays, MLUtils
using DataStructures, StatsBase, ProgressMeter, Plots

include("GPTModel.jl")
using .GPTModel

rng = Xoshiro()
Random.seed!(rng, 12345)

data = open("assets/bible.txt") do file
    content = read(file, String)
    filter(isascii, collect(content))
end

chars = sort(unique(data))
vocab_size = length(chars)
c_idx = Dict([c => i for (i, c) in enumerate(chars)])
idx_c = Dict([i => c for (i, c) in enumerate(chars)])
encode(str) = map(c -> c_idx[c], collect(str))
decode(arr) = String(map(i -> idx_c[i], arr))

data = encode(data)
n = floor(Int, length(data) * 0.8)
train_data = data[1:n]
val_data = data[n+1:end]

VOCAB_SIZE = vocab_size
BLOCK_SIZE = 128
BATCH_SIZE = 1024
EMB_SIZE = 32
ATT_SIZE = 8

function create_dataloader(data)
    x = CircularBuffer{Int32}(BLOCK_SIZE)
    y = CircularBuffer{Int32}(BLOCK_SIZE)
    xs, ys = Vector{Int32}[], Vector{Int32}[]
    for i in 1:length(data)-1
        push!(x, train_data[i])
        push!(y, train_data[i+1])
        if length(x) == BLOCK_SIZE
            push!(xs, x[:])
            push!(ys, y[:])
        end
    end
    xs = hcat(xs...)
    ys = onehotbatch(hcat(ys...), 1:VOCAB_SIZE)
    return DataLoader((xs, ys), batchsize=BATCH_SIZE, shuffle=true)
end

train_loader = create_dataloader(train_data)
val_loader = create_dataloader(val_data)
pos_enc = Int32.(collect(1:BLOCK_SIZE))
model = GPT(
    vocab_size=VOCAB_SIZE,
    block_size=BLOCK_SIZE,
    embedding_size=EMB_SIZE,
    head_size=ATT_SIZE,
    num_heads=4,
    num_blocks=2
)
ps, st = Lux.setup(rng, model)
ps = ComponentArray(ps)
opt = Optimisers.setup(Optimisers.Adam(0.0001f0), ps)
x, y = first(train_loader)
logits, st = model((x, pos_enc), ps, st)

st = Lux.trainmode(st)
for epoch in 1:10
    global model, ps, st, opt
    global losses = Float32[]
    global loss = 0.0
    @showprogress for (i, (x, y)) in enumerate(train_loader)
        grads = Zygote.gradient(ps, st) do ps, st
            logits, st = model((x, pos_enc), ps, st)
            loss = -sum(y .* logsoftmax(logits)) / size(x, 2)
        end
        opt, ps = Optimisers.update(opt, ps, grads[1])
        push!(losses, loss)
    end
    println("Epoch: $epoch, Loss: $(mean(losses))")
end

function generate(model, ps, st, prompt; max_tokens=1000, temperature=1.0)
    st = Lux.testmode(st)
    context = deepcopy(prompt)
    pos = Int32.(collect(1:BLOCK_SIZE))

    for _ in 1:max_tokens
        # Take last block_size tokens if context is too long
        x = length(context) >= BLOCK_SIZE ? context[end-BLOCK_SIZE+1:end] : context
        x = reshape(Int32.(x), :, 1)  # Make it (seq_len, batch_size=1)

        # Get model predictions
        logits, _ = model((x, pos[1:size(x, 1)]), ps, st)

        # Sample from the last position
        probs = softmax(logits[:, end, 1] ./ temperature)
        next_token = sample(1:VOCAB_SIZE, Weights(probs))
        push!(context, next_token)
    end

    return context
end

# Test generation
prompt = encode("thus saith the lord")

println("Generated Text, Temperature=1.0")
generated = generate(model, ps, st, prompt)
println(decode(generated))

println("\n\nGenerated Text, Temperature=0.9")
generated = generate(model, ps, st, prompt, temperature=0.9)
println(decode(generated))

println("\n\nGenerated Text, Temperature=0.8")
generated = generate(model, ps, st, prompt, temperature=0.8)
println(decode(generated))
