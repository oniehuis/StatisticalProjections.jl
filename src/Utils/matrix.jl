function find_invariant_and_variant_columns(M::AbstractMatrix)
    invariant_columns = Int[]
    variant_columns = Int[]

    for j in axes(M, 2)
        col = @view M[:, j]
        if all(x -> x == col[1], col)
            push!(invariant_columns, j)
        else
            push!(variant_columns, j)
        end
    end
    
    invariant_columns, variant_columns
end
