#This file contains the core functions from project 1.2
@doc """`lattice` constructs an array of lattice coordinates 
    ##### Input parameters of `lattice`:
    - `N::Int`    : Number of Points in each direction
    - `D::Int`    : Dimention of the lattice
    ##### Output values(n): 
    - `n::Array`  : D-dim Array with N points in each direction and n = 0 in the center""" ->
function lattice(N::Int, D::Int)  
    @assert(N > 0, "N = " * string(N) * " must be positive")
    @assert(D > 0, "D = " * string(D) * " must be positive")
    
    if mod(N,2) == 0 n_1D = -(N-2)/2:N/2 end
    if mod(N,2) == 1 n_1D = -(N-1)/2:(N-1)/2 end

    if D == 1 return collect(n_1D) end
    
    n = collect(Base.product([n_1D for i in 1:D]...)) #Cartesian product
    
    return n
end

@doc """`potential` calculates a dimensionless quartic potential
    ##### Input parameters of `potential`:
    - `μ::Number`    : dimensionless parameter mωr^2/ħ
    - `ϵ::Number`    : dimensionless parameter a/r with a the lattice spacing a
    - `n::Array`     : lattice coordinates
    ##### Output values(V): 
    - `V::Array`    : quartic potential μ/8 * (ϵ^2*dot(n,n) - 1)^2 """ ->
function potential(μ::Number, ϵ::Number, n::Array) 
    @assert(μ > 0, "μ = " * string(μ) * " must be positive")
    @assert(ϵ > 0, "ϵ = " * string(ϵ) * " must be positive")
    
    return @. μ/8 * (ϵ^2*dot(n,n) - 1)^2
end


@doc """`kinetic` takes ψ and returns Hψ for the free Hamiltonian with periodic boundary conditions
    ##### Input parameters of `kinetic`:
    - `μ::Number`    : dimensionless parameter mωr^2/ħ
    - `ϵ::Number`    : dimensionless parameter a/r with a the lattice spacing a
    - `ψ::Array`     : discretized wavefunction
    ##### Output values(Hψ): 
    - `Hψ::Array`    : -1/(2*μ*ϵ^2) * Δ(ψ) """ ->
function kinetic(μ::Number, ϵ::Number, ψ::Array) 
    @assert(μ > 0, "μ = " * string(μ) * " must be positive")
    @assert(ϵ > 0, "ϵ = " * string(ϵ) * " must be positive")
    
    D = ndims(ψ)
    N = size(ψ)[1]

    Δ = -2*D*ψ #last term of discretized Laplacian

    for k in 1:D
        Δ += selectdim(ψ, k, vcat(2:N,1:1)) + selectdim(ψ, k, vcat(N:N,1:N-1)) #remaining terms
    end
    
    return -1/(2*μ*ϵ^2) * Δ
end

@doc """`Hamiltonian`` takes ψ and returns Hψ, using the functions `kinetic` and `potential`
    ##### Input parameters of `kinetic`:
    - `μ::Number`    : dimensionless parameter mωr^2/ħ
    - `ϵ::Number`    : dimensionless parameter a/r with a the lattice spacing a
    - `ψ::Array`     : discretized wavefunction
    - `V::Array`     : quartic potential 
    ##### Output values(Hψ): 
    - `Hψ::Array`    : free Hamiltonian plus Potential acting on wavefunction """ ->
function Hamiltonian(μ::Number, ϵ::Number, ψ::Array, V::Array)
    @assert(size(ψ) == size(V), "ψ and V must have the same shape")
    
    return kinetic(μ, ϵ, ψ) + V .* ψ
end

@doc """`power_method` calculates an approximation of the largest eigenvalue λ
    and corresponding eigenvector ν of the positive-definite linear map 'A'. The quality
    of the approximation is dictated by the tolerance 'tol', in the sense that the
    approximated eigenvalue and eigenvector satisfy
      |A(ν) - λ*ν| < tol
    The vectors over which 'A' acts are generically d-dimensional arrays. More precisely,
    they are instances of Array with shape 'vshape'.
    The linear map 'A' is indirectly provided as a function 'apply_A' which takes a vector
    ν and returns the vector A(ν).   
    ##### Input parameters of `power_method`:
    - `vshape::Tuple{Vararg{Int}}` : shape of the arrays over which 'A' acts
    - `apply_A::Function` : function ν -> A(ν)
    - `tol::Float64`  : tolerance
    - `maxiters::Int` : maximum number of iterations
    ##### optional input parameter 
    - `init::Array=rand(ComplexF64, vshape)` : initial value for ν 
    ##### Output values: (λ, ν, niters)
    - `λ::Float64` : largest eigenvalue of A
    - `ν::Array`    : corresponding eigenvector
    - `niters::Int` : number of iterations""" ->
function power_method(vshape::Tuple{Vararg{Int}}, apply_A::Function, tol::Float64, maxiters::Int; init::Array=rand(ComplexF64, vshape))
    @assert(tol > 0, "tol = " * string(tol) * " must be positive")
    @assert(maxiters > 0, "maxiters = " * string(maxiters) * " must be positive")
    @assert(size(init) == vshape, "init must have size vshape = " * string(vshape))
    
    v = init
    niters = 0
    λ = 0
    
    while niters < maxiters
        w = apply_A(v)
        λ = norm(w)
        if max(abs.(w - λ*v)...) < tol break end
        v = w/λ
        niters += 1
    end
    
    if niters == maxiters error("Maximum number of power method iterations reached") end
    
    return λ, v, niters
end

@doc """`conjugate_gradient` calculates an approximation 'x' of 'A^{-1}(b)', where
 'A' is a positive-definite linear map 'A', and 'b' is a vector in the domain of 'A'.
 The quality of the approximation is dictated by the tolerance 'tol', in the sense
 that the following inequality is satisfied |A(x) - b| <= tol |b|.
 The vectors over which 'A' acts are generically d-dimensional arrays. More precisely,
 they are instances of 'Array' with shape 'vshape'.
 The linear map 'A' is indirectly provided as a function 'apply_A' which takes a vector
 ν and returns the vector A(ν).

##### Input parameters of `conjugate_gradient`:
- `apply_A::Function` : function ν -> A(ν)
- `b::Array`      : vector 'b'
- `tol::Float64`  : tolerance
- `maxiters::Int` : maximum number of iterations
##### Output values: (x, niters)
- `x::Array`      : approximation of 'A^{-1}(b)'
 -`niters::Int`  : number of iterations""" ->
function conjugate_gradient(apply_A::Function, b::Array, tol::Float64, maxiters::Int)
    @assert(tol > 0, "tol = " * string(tol) * " must be positive")
    @assert(maxiters > 0, "maxiters = " * string(maxiters) * " must be positive")
    
    x = rand(ComplexF64, size(b))
    r = b - apply_A(x); p = r
    niters = 0
    
    while niters < maxiters
        x = x + dot(r,r) / dot(p,apply_A(p)) * p
        r_new = b - apply_A(x)
        if norm(r_new) < tol*norm(b) break end
        p = r_new + dot(r_new,r_new) / dot(r,r) * p
        r = r_new
        niters += 1
    end
    
    if niters == maxiters error("Maximum number of conjugate gradient iterations reached") end
    
    return x, niters
end

@doc """`lowest_eigenvalue` calculates the smallest eigenvalue of A and the corresponding eigenvector, 
using `conjugate_gradient` and `power_method`
##### Input parameters of `lowest_eigenvalue`:
- `vshape::Tuple{Vararg{Int}}` : shape of the arrays over which 'A' acts
- `apply_A::Function`: function ν -> A(ν)
- `tol_pm::Float64`  : tolerance of the `power_method`
- `maxiters_pm::Int` : maximum number of iterations of the `power_method`
- `tol_cg::Float64`  : tolerance of the `conjugate_gradient`
- `maxiters_cg::Int` : maximum number of iterations of the `conjugate_gradient`
##### optional input parameter 
- `init::Array=rand(ComplexF64, vshape)` : initial value for ν 
##### Output values: (1/λ, v)
- `1/λ`: lowest eigenvalue
- `v`  : corresponding eigenvector""" ->
function lowest_eigenvalue(vshape::Tuple{Vararg{Int}}, apply_A::Function, tol_pm::Float64, maxiters_pm::Int, tol_cg::Float64, maxiters_cg::Int; init::Array=rand(ComplexF64, vshape))
    
    function apply_A_inverse(v)
        x, _ = conjugate_gradient(apply_A, v, tol_cg, maxiters_cg)
        return x
    end
    
    λ, v, _ = power_method(vshape, apply_A_inverse, tol_pm, maxiters_pm; init = init)
    
    return 1/λ, v
end

@doc """`energy_expectation` calculates the energy expectation value and variance of the wave function ψ  
using the functions `lattice`, `potential` and `Hamiltonian`
##### Input parameters of `energy_expectation`:
- `ψ::Array`     : discretized wavefunction  
- `μ::Number`    : dimensionless parameter mωr^2/ħ
- `ϵ::Number`    : dimensionless parameter a/r with a the lattice spacing a
##### Output values: (exp_E, var_E)
- `exp_E::Number`: energy expectation value
- `√|(var_E)|::Number`  : energy variance""" ->
function energy_expectation(ψ::Array, μ::Number, ϵ::Number)
    n = lattice(size(ψ)[1], ndims(ψ))
    V = potential(μ, ϵ, n)

    exp_E = real(dot(ψ, Hamiltonian(μ, ϵ, ψ, V)))
    var_E = real(dot(Hamiltonian(μ, ϵ, ψ, V), Hamiltonian(μ, ϵ, ψ, V))) - exp_E^2
    
    return exp_E, sqrt(abs(var_E))
end

@doc """`position_operator` multiplies the positions of points on the lattice componentwise with ψ
##### Input parameter of `position_operator`:
- `ψ::Array`     : discretized wavefunction  
##### Output values: (rψ)
- `rψ::Array`: position operator applied to ψ """ ->
function position_operator(ψ::Array)
    D = ndims(ψ)
    n = lattice(size(ψ)[1],D)
    
    rψ = map(x -> x[1], n) .* ψ
    for k in 2:D
        rψ = cat(rψ, map(x -> x[k], n) .* ψ, dims = D+1)
    end
    
    return rψ
end

@doc """`momentum_operator` multiplies the differnce between neighbors on the lattice componentwise with -i/2*ψ
##### Input parameter of `momentum_operator`:
- `ψ::Array`     : discretized wavefunction  
##### Output values: (-i/2*pψ)
- `-i/2*pψ::Array`: momentum operator applied to ψ """ ->
function momentum_operator(ψ::Array)
    N = size(ψ)[1]
    D = ndims(ψ)
    
    pψ = selectdim(ψ, 1, vcat(2:N,1:1)) - selectdim(ψ, 1, vcat(N:N,1:N-1))
    for k in 2:D
        pψ = cat(pψ, (selectdim(ψ, k, vcat(2:N,1:1)) - selectdim(ψ, k, vcat(N:N,1:N-1))), dims = D+1)
    end
    
    return -im/2 * pψ
end

@doc """`position_expectation` calculates the position expectation value and variance of the wave function ψ  
using the function `position_operator`
##### Input parameter of power_method:
- `ψ::Array`     : discretized wavefunction  
##### Output values: (exp_r, var_r)
- `exp_r::Number`: position expectation value
- `√|(var_r)|::Number`  : position variance""" ->
function position_expectation(ψ::Array)
    rψ = position_operator(ψ)
    D = ndims(ψ)

    exp_r = Tuple(real(dot(ψ, selectdim(rψ, D+1, k))) for k in 1:D)
    var_r = Tuple(real(dot(selectdim(rψ, D+1, k), selectdim(rψ, D+1, k))) for k in 1:D) .- exp_r.^2

    return exp_r, sqrt.(var_r)
end

@doc """`momentum_expectation` calculates the momentum expectation value and variance of the wave function ψ  
using the function `momentum_operator`
##### Input parameter of `momentum_expectation`:
- `ψ::Array`     : discretized wavefunction  
##### Output values: (exp_p, var_p)
- `exp_p::Number`: momentum expectation value
- `√|(var_p)|::Number`  : momentum variance""" ->
function momentum_expectation(ψ::Array)
    pψ = momentum_operator(ψ)
    D = ndims(ψ)

    exp_p = Tuple(real(dot(ψ, selectdim(pψ, D+1, k))) for k in 1:D)
    var_p = Tuple(real(dot(selectdim(pψ, D+1, k), selectdim(pψ, D+1, k))) for k in 1:D) .- exp_p.^2

    return exp_p, sqrt.(var_p)
end

@doc """`halfspace_prob` calculates the probability to find the particle in the half-line x > 0.
##### Input parameter of `halfspace_prob`:
- `ψ::Array`  : discretized wavefunction  
##### Output values: (p)
- `p::Number` : probability""" ->
function halfspace_prob(ψ::Array)
    N = size(ψ)[1]
    D = ndims(ψ)

    if mod(N,2) == 0 m = round(Int, (N-2)/2) end
    if mod(N,2) == 1 m = round(Int, (N-1)/2) end
    ind = [m+2:N for k in 1:D]
    
    return real(dot(ψ[ind...], ψ[ind...]))
end

@doc """### The `ground_state` function
1. reads the relevant parameters from an input file or from command line,
2. calculates the lowest eigenvalue and eigenvector of the Hamiltonian given in the notes using the modules defined above,
3. writes the eigenvector to file (so that the wavefunction can be plotted at a later time),
4. calculates and prints all observables defined in the observable module.calculates the smallest eigenvalue of A and the corresponding eigenvector, 
using all functions in this file
##### Input parameters of `ground_state`:
- `shape::Tuple{Vararg{Int}}` : shape of the arrays over which 'A' acts
- `μ::Number`    : dimensionless parameter mωr^2/ħ
- `ϵ::Number`    : dimensionless parameter a/r with a the lattice spacing a
- `tol_pm::Float64`  : tolerance of the `power_method`
- `maxiters_pm::Int` : maximum number of iterations of the `power_method`
- `tol_cg::Float64`  : tolerance of the `conjugate_gradient`
- `maxiters_cg::Int` : maximum number of iterations of the `conjugate_gradient`
##### optional input parameter 
- `init::Array=rand(ComplexF64, vshape)` : initial value for ν 
- `verbose::Bool=true`: if true function also calculates the position and momentum expectation value
##### Output values: (E_min, ψ_min)
- `E_min`: groundstate eigenvalue
- `ψ_min:Array`  : groundstate eigenvector""" ->
function ground_state(shape::Tuple{Vararg{Int}}, μ::Number, ϵ::Number, tol_pm::Float64, maxiters_pm::Int, tol_cg::Float64, maxiters_cg::Int; init::Array=rand(ComplexF64, shape), verbose::Bool=true)
    D = length(shape)
    n = lattice(shape[1], D)
    V = potential(μ, ϵ, n)
    
    function apply_H(ψ)
        return Hamiltonian(μ, ϵ, ψ, V)
    end
    
    E_min, ψ_min = lowest_eigenvalue(shape, apply_H, tol_pm, maxiters_pm, tol_cg, maxiters_cg; init = init)
    ψ_min = ψ_min/norm(ψ_min)
    
    if verbose
    r = position_expectation(ψ_min)
    p = momentum_expectation(ψ_min)
    fmt = Printf.Format("(" * "%.5e, "^(D-1) * "%.5e)")
    
    @printf "Energy: %.5e ± %.5e\n" energy_expectation(ψ_min, μ, ϵ)...
    println("Position: ", Printf.format(fmt, r[1]...), " ± ", Printf.format(fmt, r[2]...))
    println("Momentum: ", Printf.format(fmt, p[1]...), " ± ", Printf.format(fmt, p[2]...))
    @printf "Half-space probability: %.5e" halfspace_prob(ψ_min)
    #println("\nΔxΔp: ", Printf.format(fmt, (r[2] .* p[2])...)) #only consistently ≥ 1/2 if everything is sufficiently smooth
    end

    return E_min, ψ_min
end