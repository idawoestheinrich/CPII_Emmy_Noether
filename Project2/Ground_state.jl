#This file contains the core functions from project 1.2

function lattice(N::Int, D::Int) #construct array of lattice coordinates (with n = 0 in the center)
    @assert(N > 0, "N = " * string(N) * " must be positive")
    @assert(D > 0, "D = " * string(D) * " must be positive")
    
    if mod(N,2) == 0 n_1D = -(N-2)/2:N/2 end
    if mod(N,2) == 1 n_1D = -(N-1)/2:(N-1)/2 end

    if D == 1 return collect(n_1D) end
    
    n = collect(Base.product([n_1D for i in 1:D]...)) #Cartesian product
    
    return n
end

function potential(μ::Number, ϵ::Number, n::Array)
    @assert(μ > 0, "μ = " * string(μ) * " must be positive")
    @assert(ϵ > 0, "ϵ = " * string(ϵ) * " must be positive")
    
    return @. μ/8 * (ϵ^2*dot(n,n) - 1)^2
end

function kinetic(μ::Number, ϵ::Number, ψ::Array) #with periodic boundary conditions
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

function Hamiltonian(μ::Number, ϵ::Number, ψ::Array, V::Array)
    @assert(size(ψ) == size(V), "ψ and V must have the same shape")
    
    return kinetic(μ, ϵ, ψ) + V .* ψ
end

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

function lowest_eigenvalue(vshape::Tuple{Vararg{Int}}, apply_A::Function, tol_pm::Float64, maxiters_pm::Int, tol_cg::Float64, maxiters_cg::Int; init::Array=rand(ComplexF64, vshape))
    
    function apply_A_inverse(v)
        x, _ = conjugate_gradient(apply_A, v, tol_cg, maxiters_cg)
        return x
    end
    
    λ, v, _ = power_method(vshape, apply_A_inverse, tol_pm, maxiters_pm; init = init)
    
    return 1/λ, v
end

function energy_expectation(ψ::Array, μ::Number, ϵ::Number)
    n = lattice(size(ψ)[1], ndims(ψ))
    V = potential(μ, ϵ, n)

    exp_E = real(dot(ψ, Hamiltonian(μ, ϵ, ψ, V)))
    var_E = real(dot(Hamiltonian(μ, ϵ, ψ, V), Hamiltonian(μ, ϵ, ψ, V))) - exp_E^2
    
    return exp_E, sqrt(abs(var_E))
end

function position_operator(ψ::Array)
    D = ndims(ψ)
    n = lattice(size(ψ)[1],D)
    
    rψ = map(x -> x[1], n) .* ψ
    for k in 2:D
        rψ = cat(rψ, map(x -> x[k], n) .* ψ, dims = D+1)
    end
    
    return rψ
end

function momentum_operator(ψ::Array)
    N = size(ψ)[1]
    D = ndims(ψ)
    
    pψ = selectdim(ψ, 1, vcat(2:N,1:1)) - selectdim(ψ, 1, vcat(N:N,1:N-1))
    for k in 2:D
        pψ = cat(pψ, (selectdim(ψ, k, vcat(2:N,1:1)) - selectdim(ψ, k, vcat(N:N,1:N-1))), dims = D+1)
    end
    
    return -im/2 * pψ
end

function position_expectation(ψ::Array)
    rψ = position_operator(ψ)
    D = ndims(ψ)

    exp_r = Tuple(real(dot(ψ, selectdim(rψ, D+1, k))) for k in 1:D)
    var_r = Tuple(real(dot(selectdim(rψ, D+1, k), selectdim(rψ, D+1, k))) for k in 1:D) .- exp_r.^2

    return exp_r, sqrt.(var_r)
end

function momentum_expectation(ψ::Array)
    pψ = momentum_operator(ψ)
    D = ndims(ψ)

    exp_p = Tuple(real(dot(ψ, selectdim(pψ, D+1, k))) for k in 1:D)
    var_p = Tuple(real(dot(selectdim(pψ, D+1, k), selectdim(pψ, D+1, k))) for k in 1:D) .- exp_p.^2

    return exp_p, sqrt.(var_p)
end

function halfspace_prob(ψ::Array)
    N = size(ψ)[1]
    D = ndims(ψ)

    if mod(N,2) == 0 m = round(Int, (N-2)/2) end
    if mod(N,2) == 1 m = round(Int, (N-1)/2) end
    ind = [m+2:N for k in 1:D]
    
    return real(dot(ψ[ind...], ψ[ind...]))
end

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