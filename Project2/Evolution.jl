#This file contains the core functions from project 1.3
#Requires Ground_state.jl and FFTW!

function wave_packet(n::Array, m::Tuple, σ::Number, k::Tuple) #centered around m, variance σ, momentum k
    @assert(length(m) == ndims(n), "m = " * string(m) * " must have the correct dimension")

    D = ndims(n)
    X = [i .- m for i in n]
    K = fill(k, size(n))
    
    return @. 1/(sqrt(π)*σ)^(D/2) * exp(-dot(X,X) / (2*σ^2)) * exp(im*dot(K,n))
end

function euler(μ::Number, ϵ::Number, ψ::Array, V::Array, τ::Number)
    @assert(τ > 0, "τ = " * string(τ) * " must be positive")
    
    return ψ - im*τ * Hamiltonian(μ, ϵ, ψ, V)
end

function crank_nicolson(μ::Number, ϵ::Number, ψ::Array, V::Array, τ::Number, tol_cg::Number, maxiters_cg::Int)
    @assert(τ > 0, "τ = " * string(τ) * " must be positive")

    function apply_H(ψ)
        return Hamiltonian(μ, ϵ, ψ, V)
    end

    function apply_A(ψ)
        return ψ + τ^2/4 * apply_H(apply_H(ψ))
    end

    η, _ = conjugate_gradient(apply_A, ψ, tol_cg, maxiters_cg)
    return η - im*τ * apply_H(η) - τ^2/4 * apply_H(apply_H(η))
end

function strang_splitting(μ::Number, ϵ::Number, ψ::Array, V::Array, τ::Number, plan_fft, plan_ifft) #type of plan_fft?
    @assert(τ > 0, "τ = " * string(τ) * " must be positive")

    D = ndims(ψ)
    
    k_1D = 0:N-1
    k = collect((Base.product([k_1D for i in 1:D]...)))
    sin_k = map(x -> (@. sin(pi/N * x)^2), k)
    K_ft = 2/(μ*ϵ^2) * sum.(sin_k)

    η = @. exp(-im*τ/2 * V) * ψ
    η_ft = plan_fft * η
    ξ_ft = @. exp(-im*τ * K_ft) * η_ft
    ξ = plan_ifft * ξ_ft

    return @. exp(-im*τ/2 * V) * ξ
end

function choose_integrator(method; tol_cg::Number=1e-10, maxiters_cg::Int=10000)
    methods = ["euler", "crank-nicolson", "strang-splitting"]
    @assert(method in methods, "method = " * string(method) * " must be one of " * string(methods))

    if method == methods[1]
        out = (μ, ϵ, ψ, V, τ) -> euler(μ, ϵ, ψ, V, τ)
    elseif method == methods[2]
        out = (μ, ϵ, ψ, V, τ) -> crank_nicolson(μ, ϵ, ψ, V, τ, tol_cg, maxiters_cg)
    elseif method == methods[3]
        P = plan_fft(ψ_0); P_inv = inv(P) #is this significantly quicker?
        out = (μ, ϵ, ψ, V, τ) -> strang_splitting(μ, ϵ, ψ, V, τ, P, P_inv)
    end
    
    return out
end

function simulate_1D(N::Int, μ::Number, ϵ::Number, T::Number, τ::Number, ψ_0::Array, method::String, fps::Int, gif_length::Number; tol_cg::Number=1e-10, maxiters_cg::Int=10000)
    @assert(T > 0, "T = " * string(T) * " must be positive")
    @assert(fps > 0, "fps = " * string(fps) * " must be positive")
    @assert(gif_length > 0, "gif_length = " * string(gif_length) * " must be positive")
    
    integrator = choose_integrator(method; tol_cg, maxiters_cg)
    
    n = lattice(N, 1)
    V = potential(μ, ϵ, n)
    ψ = ψ_0; t = 0
    
    anim = @animate while t < T
        ψ = integrator(μ, ϵ, ψ, V, τ); t += τ
        fmt = Printf.Format("t = %.1e")
        plot(n, abs.(ψ), label = Printf.format(fmt, t))
        ylims!(0, 1)
        plot!(n, V/max(V...), color = 2, label = "∝ V(n)")
        vline!([-1/ϵ, 1/ϵ], color = 2, label = "", linestyle = :dash)
        #title!("m = " * string(round.(Int, m)) * ", k = " *string(k) * ", σ = " * string(σ))
    end every round(Int, 1/(gif_length*fps) * T/τ)

    gif(anim, method * ".gif", fps = fps)
end

function simulate_2D(N::Int, μ::Number, ϵ::Number, T::Number, τ::Number, ψ_0::Array, method::String, fps::Int, gif_length::Number; tol_cg::Number=1e-10, maxiters_cg::Int=10000)
    @assert(T > 0, "T = " * string(T) * " must be positive")
    @assert(fps > 0, "fps = " * string(fps) * " must be positive")
    @assert(gif_length > 0, "gif_length = " * string(gif_length) * " must be positive")

    integrator = choose_integrator(method; tol_cg, maxiters_cg)
    
    n_1D = lattice(N, 1)
    n = lattice(N, 2)
    V = potential(μ, ϵ, n)
    ψ = ψ_0; t = 0
    
    anim = @animate while t < T
        ψ = integrator(μ, ϵ, ψ, V, τ); t += τ
        fmt = Printf.Format("t = %.1e")
        surface(n_1D, n_1D, abs.(ψ), cmap = :blues, title = Printf.format(fmt, t), legend = nothing)
        zlims!(0, 0.1)
        #surface!(n_1D, n_1D, V/max(V...) * 0.1, cmap = :heat, alpha = :0.3, legend = nothing)
        #plot sircle of minima in the x-y-plane?
    end every round(Int, 1/(gif_length*fps) * T/τ)

    gif(anim, method * "_2D.gif", fps = fps)
end