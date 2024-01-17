#This file contains the core functions from project 2.1
#Requires LinearAlgebra,Random,Statistics,Dates,Printf!

function s_init(N, D, type)
    types = ["cold", "hot"]
    @assert(type in types, "type must be one of " * string(types))
    
    if type == "cold" return fill(1, Tuple(N for k in 1:D))
    else return rand([-1,1], Tuple(N for k in 1:D)) end
end

function Hamiltonian(β::Number, b::Number, s::Array{Int64})
    S = sum([selectdim(s, k, vcat(2:N,1:1)) for k in 1:D])
    return -β * sum(s .* S) - b * sum(s)
end

function Metropolis(β::Number, b::Number, s::Array{Int64})
    D = ndims(s)
    N = size(s)[1]

    for i in CartesianIndices(s)
        ΔH = b*s[i]
        for k in 1:D
            i_plus = [Tuple(i)...]; i_plus[k] = mod1(i[k]+1, N)
            i_minus = [Tuple(i)...]; i_minus[k] = mod1(i[k]-1, N)
            ΔH += β*s[i] * (s[CartesianIndex(i_plus...)] + s[CartesianIndex(i_minus...)])
        end
        if exp(-2*ΔH) > rand(Float64) s[i] = -s[i] end
    end
    
    return s
end

function Metropolis_v2(β::Number, b::Number, s::Array{Int64})
    D = ndims(s)
    N = size(s)[1]
    
    #for each grid pt n, calculate the total spin of all pts of the form (n ± e_k) for some k:
    s_plus = sum([circshift(s, 1*Matrix(I, D, D)[k,:]) for k in 1:D])
    s_minus = sum([circshift(s, -1*Matrix(I, D, D)[k,:]) for k in 1:D])

    for i in CartesianIndices(s)
        ΔH = β*s[i] * (s_plus[i] + s_minus[i]) + b*s[i]
        if exp(-2*ΔH) > rand(Float64)
            s[i] = -s[i]
            for k in 1:D #update s_plus & s_minus
                i_plus = [Tuple(i)...]; i_plus[k] = mod1(i[k]+1, N)
                i_minus = [Tuple(i)...]; i_minus[k] = mod1(i[k]-1, N)
                s_plus[CartesianIndex(i_minus...)] += 2*s[i]
                s_minus[CartesianIndex(i_plus...)] += 2*s[i]
            end
        end
    end
    
    return s
end

function Markov_chain(β::Number, b::Number, N::Number, D::Number, init::String, N_config::Int)
    @assert(β > 0, "β = " *string(β) * " must be positive")
    @assert(b > 0, "b = " *string(b) * " must be positive")
    @assert(N_config > 0, "N_config = " * string(N_config) * " must be positive")

    s = s_init(N, D, init)
    M, E = [sum(s)], [Hamiltonian(β, b, s)]
    
    for m in 1:N_config
        #s = Metropolis(β, b, s)
        s = Metropolis_v2(β, b, s)
        M = vcat(M, sum(s))
        E = vcat(E, Hamiltonian(β, b, s))
    end
    
    return M, E
end