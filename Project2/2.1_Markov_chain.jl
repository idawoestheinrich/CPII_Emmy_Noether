#This file contains the core functions from project 2.1
#Requires LinearAlgebra,Random,Statistics,Dates,Printf!
@doc """`s_init` constructs the initial spin configuration on a D dimensionall lattice with N particles in each direction.
    ##### Input parameters of `s_init`:
    - `N::Int`    : Number of particles in each direction
    - `D::Int`    : Dimention of the lattice
    - `type::string` : Must be one of these
        - `"cold"` :  All spins equal to +1
        - `"hot"`  :  Randomly generated spin configuration (each spin has 50% probability to be +1 or -1)
    ##### Output values(n): 
    - `s::Array`  : D-dim Array with N Spins of +1 or -1 in each direction""" ->
function s_init(N, D, type)
    types = ["cold", "hot"]
    @assert(type in types, "type must be one of " * string(types))
    
    if type == "cold" return fill(1, Tuple(N for k in 1:D))
    else return rand([-1,1], Tuple(N for k in 1:D)) end
end

@doc raw"""`Hamiltonian` calculates the energy of a spin configuration.
    ```math
    H(s) = -J\sum_n \sum_{k=1}^D s(\underline{n})s(\underline{n}+\underline{e}_k) - B\mu\sum_n s(\underline{n})
    ```
    whereby the first term corresponds to the nearest neighbor interaction and the second to the interaction with an external magnetic field b.
    ##### Input parameters of `Hamiltonian`:
    - `β::Number` : β = J/kT dimensionless coupling constant
    - `b::Number` : b = μB/kT dimensionless external magnetic field
    - `s::Array`  : D-dim Array with N Spins of +1 or -1 in each direction
    ##### Output values(n): 
    - `H::Number` : Energy value as described above
    """ ->
function Hamiltonian(β::Number, b::Number, s::Array{Int64})
    S = sum([circshift(s, 1*Matrix(I, D, D)[k,:]) for k in 1:D])
    return -β * sum(s .* S) - b * sum(s)
end

@doc """`Metropolis` algorithm constructs a new spin configuration. When appied multiple times it can be used in a `Markov_chain` (see function below).
    ##### Input parameters of `Metropolis`:
    - `β::Number` : β = J/kT dimensionless coupling constant
    - `b::Number` : b = μB/kT dimensionless external magnetic field
    - `s::Array`  : D-dim Array with N Spins of +1 or -1 in each direction
    ##### Output values(n): 
    - `s::Array`  : D-dim Array with new configuration of the N Spins in each direction""" ->
function Metropolis(β::Number, b::Number, s::Array{Int64})
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
                i_plus = ntuple(j -> j==k ? mod1(i[j] + 1, N) : i[j], D)
                i_minus = ntuple(j -> j==k ? mod1(i[j] - 1, N) : i[j], D)
                s_minus[CartesianIndex(i_plus)] += 2*s[i]
                s_plus[CartesianIndex(i_minus)] += 2*s[i]
            end
        end
    end
    
    return s
end

@doc raw"""`Markov_chain` Monte Carlo algorithm generates a stochastic sequence of magnetisation and energy configurations by iterating over the `Metropolis` function.
    The thermalized configuration is distributed with the probability
    ```math
    P(s) = \frac{1}{Z}e^{-\frac{H(s)}{k_B T}}
    ```
    This function needs `s_init`, `Hamiltonian` and `Metropolis` to work.
    ##### Input parameters of `Markov_chain`:
    - `β::Number` : β = J/kT dimensionless coupling constant
    - `b::Number` : b = μB/kT dimensionless external magnetic field
    - `N::Int`    : Number of particles in each direction
    - `D::Int`    : Dimension of the lattice
    - `init::string` : Type of initial spin configuration. Must be one of these
        - `"cold"` :  All spins equal to +1
        - `"hot"`  :  Randomly generated spin configuration (each spin has 50% probability to be +1 or -1)
    - `N_config::Int`: Number of configurations
    ##### Output values(n): 
    - `M::Vector`  : Vector of length `N_config` containing the magnetisation for each configuration
    - `E::Vector`  : Vector of length `N_config` containing the Energy for each configuration""" ->
function Markov_chain(β::Number, b::Number, N::Number, D::Number, init::String, N_config::Int)
    @assert(β > 0, "β = " *string(β) * " must be positive")
    @assert(b > 0, "b = " *string(b) * " must be positive")
    @assert(N_config > 0, "N_config = " * string(N_config) * " must be positive")
    
    s = s_init(N, D, init)
    M, E = [sum(s)], [Hamiltonian(β, b, s)]
        
    for m in 1:N_config
        #s = Metropolis(β, b, s)
        s = Metropolis(β, b, s)
        M = vcat(M, sum(s))
        E = vcat(E, Hamiltonian(β, b, s))
    end
        
    return M, E
end

function replica_method(β::Number, b::Number, N::Number, D::Number, init::String, N_config::Int, N_rep::Int, cutoff::Int;random::Bool=false verbose::Bool=true)
    @assert(N_rep > 0, "N_rep = " * string(N_rep) * " must be positive")
    @assert(cutoff >= 0 && cutoff < N_config, "cutoff = " * string(cutoff) * " must be between 0 and N_config")
    
    M_bar, E_bar = [], []
    for r in 1:N_rep
        if random Random.seed!(Dates.value(Millisecond(now())))
        else Random.seed!(r) end 
        M, E = Markov_chain(β, b, N, D, init, N_config)
        M_bar = vcat(M_bar, mean(M[cutoff:end]))
        E_bar = vcat(E_bar, mean(E[cutoff:end]))
    end

    M_est, E_est = mean(M_bar), mean(E_bar)
    M_err = sqrt(1/(N_rep*(N_rep-1)) * sum((M_bar .- M_est).^2))
    E_err = sqrt(1/(N_rep*(N_rep-1)) * sum((E_bar .- E_est).^2))

    if verbose
        @printf "⟨M⟩ = %.5e ± %.5e\n" (M_est, M_err)...
        @printf "⟨E⟩ = %.5e ± %.5e" (E_est, E_err)...
    end
    
    return (M_est, M_err), (E_est, E_err)
end