module GPTModel

export GPT

using Lux, ComponentArrays, LinearAlgebra

function causal_self_attention(k, q, v, scale, mask)
    attention_scores = batched_mul(permutedims(q, (2, 1, 3)), k) .* scale  # (T, T, B)
    masked_scores = attention_scores .+ mask
    attention_weights = softmax(masked_scores; dims=2)
    return batched_mul(v, permutedims(attention_weights, (2, 1, 3)))
end

function create_attention_masks(block_size)
    mask = fill(-Inf32, (block_size, block_size))
    mask[tril(ones(Bool, block_size, block_size))] .= 0.0f0
    return mask
end

function create_attention_head(embedding_size, head_size, block_size)
    key = Dense(embedding_size => head_size, use_bias=false)
    query = Dense(embedding_size => head_size, use_bias=false)
    value = Dense(embedding_size => head_size, use_bias=false)
    scale = Float32(1 / sqrt(head_size))
    mask = create_attention_masks(block_size)

    return Parallel(
        (k, q, v) -> causal_self_attention(k, q, v, scale, mask),
        key, query, value
    )
end

function create_attention_block(embedding_size, head_size, num_heads, block_size)
    attention = if num_heads == 1
        Chain(
            create_attention_head(embedding_size, head_size, block_size),
            Dense(head_size => embedding_size)
        )
    else
        heads = [create_attention_head(embedding_size, head_size, block_size) for _ in 1:num_heads]
        Chain(
            Parallel((x...) -> cat(x..., dims=1), heads...),
            Dense(head_size * num_heads => embedding_size)
        )
    end
    attention_residual = Chain(SkipConnection(attention, +), LayerNorm((embedding_size, block_size)))
    ffn = Chain(
        Dense(embedding_size => 4 * embedding_size, gelu),
        Dense(4 * embedding_size => embedding_size)
    )
    ffn_residual = Chain(SkipConnection(ffn, +), LayerNorm((embedding_size, block_size)))
    return Chain(attention_residual, ffn_residual)
end

function create_embedding_layer(vocab_size, block_size, embedding_size)
    token_embedding = Embedding(vocab_size => embedding_size)
    position_embedding = Embedding(block_size => embedding_size)
    return Parallel((t, p) -> t .+ p, token_embedding, position_embedding)
end

function GPT(; vocab_size, block_size, embedding_size, head_size, num_heads=1, num_blocks=1)
    embedding = create_embedding_layer(vocab_size, block_size, embedding_size)
    attention = [create_attention_block(embedding_size, head_size, num_heads, block_size) for _ in 1:num_blocks]
    return Chain(
        embedding,
        attention...,
        Dense(embedding_size => vocab_size)
    )
end

end # module
