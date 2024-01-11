#This file contains the core functions from project 1.3
#Requires 1.2_Ground_state.jl and FFTW!

@doc """`wave_packet` generates the initial wave function as a gaussian with a certain momentum: 
    -the center , width and momentum of the gaussian should be given as parameters 
    ##### Input parameters of `wave_packet`:
    - `n::Array`    : Number of Points in each direction
    - `m::Tuple`    : the center of the wave function 
    - `σ::Number`   : width/variance 
    - `k::Tuple`    : momentum 
    ##### Output values(ψ): 
    - `ψ::Array`  : gaussian wave packet""" ->
function wave_packet(n::Array, m::Tuple, σ::Number, k::Tuple) #centered around m, variance σ, momentum k
    @assert(length(m) == ndims(n), "m = " * string(m) * " must have the correct dimension")

    D = ndims(n)
    X = [i .- m for i in n]
    K = fill(k, size(n))
    
    return @. 1/(sqrt(π)*σ)^(D/2) * exp(-dot(X,X) / (2*σ^2)) * exp(im*dot(K,n))
end

@doc """`euler` integrator approximates the next time step of the exponential exp(-iτH) with its first-order Taylor expansion exp(-iτH)→ 1-iτH
using the function `Hamiltonian`
    ##### Input parameters of `euler`:
    - `μ::Number`  : dimensionless parameter mωr^2/ħ
    - `ϵ::Number`  : dimensionless parameter a/r with a the lattice spacing a 
    - `ψ::Array`   : discretized wavefunction (q)
    - `V::Array`   : quartic potential 
    - `τ::Number`  : time step  
    ##### Output values(n): 
    - `ψ::Array`  : discretized wavefunction (q+1)""" ->
function euler(μ::Number, ϵ::Number, ψ::Array, V::Array, τ::Number)
    @assert(τ > 0, "τ = " * string(τ) * " must be positive")
    
    return ψ - im*τ * Hamiltonian(μ, ϵ, ψ, V)
end

@doc raw"""`crank_nicolson` integrator approximates the next time step of the exponential exp(-iτH) with its first-order Taylor expansion 
    ```math
    e^{-i\tau H} \rightarrow \frac{1-\frac{i}{2}\tau H}{1+\frac{i}{2}\tau H} = \frac{(1-\frac{i}{2}\tau H)^2}{1+\frac{1}{4}\tau^2 H^2}
    ``` using the functions `Hamiltonian` and `conjugate_gradient`
    ##### Input parameters of `crank_nicolson`:
    - `μ::Number`  : dimensionless parameter mωr^2/ħ
    - `ϵ::Number`  : dimensionless parameter a/r with a the lattice spacing a 
    - `ψ::Array`   : discretized wavefunction (q)
    - `V::Array`   : quartic potential 
    - `τ::Number`  : time step  
    - `tol_cg::Float64`  : tolerance of the `conjugate_gradient`
    - `maxiters_cg::Int` : maximum number of iterations of the `conjugate_gradient`    
    ##### Output values(n): 
    - `ψ::Array`  : discretized wavefunction (q+1)""" ->
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

@doc raw"""`strang_splitting` integrator approximates the next time step of the exponential exp(-iτH) by the decomposition of the Hamiltonian H = K + V in kinetic and potential energy
    ```math
    e^{-i\tau H} \rightarrow e^{-\frac{i}{2}\tau V}e^{-i\tau K}e^{-\frac{i}{2}\tau V}
    ``` 
    ##### Input parameters of `strang_splitting`:
    - `μ::Number`  : dimensionless parameter mωr^2/ħ
    - `ϵ::Number`  : dimensionless parameter a/r with a the lattice spacing a 
    - `ψ::Array`   : discretized wavefunction (q)
    - `V::Array`   : quartic potential 
    - `τ::Number`  : time step
    - `plan_fft`   : pre-computed fast fourier transform (FFT) see https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.plan_fft
    - `plan_ifft`  : pre-computed inverse FFT see https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.plan_ifft   
    ##### Output values(ψ): 
    - `ψ::Array`  : discretized wavefunction (q+1)""" ->
function strang_splitting(μ::Number, ϵ::Number, ψ::Array, V::Array, τ::Number, plan_fft, plan_ifft)
    @assert(τ > 0, "τ = " * string(τ) * " must be positive")
    type_fft = FFTW.cFFTWPlan{ComplexF64, -1, false, ndims(ψ), Tuple{[Int64 for i in 1:ndims(ψ)]...}}
    type_ifft = AbstractFFTs.ScaledPlan{ComplexF64, FFTW.cFFTWPlan{ComplexF64, 1, false, ndims(ψ), Tuple{[Int64 for i in 1:ndims(ψ)]...}}, Float64}
    @assert(typeof(plan_fft) == type_fft, "plan_fft must be a pre-planned FFT for an array of size " * string(size(ψ)))
    @assert(typeof(plan_ifft) == type_ifft, "plan_ifft must be a pre-planned inverse FFT for an array of size " * string(size(ψ)))
    
    N = size(ψ)[1]
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

@doc """`choose_integrator` returns the chosen the integrator function, it is necessary, because `if` does not create a local scope in julia
    ##### Input parameters of `choose_integrator`:
    - `method::string` : one of the methods, `euler`, `crank-nicolson` or  `strang-splitting`
    ##### Optional parameters:
    - `tol_cg::Number`   : tolerance of the `conjugate_gradient`
    - `maxiters_cg::Int` : maximum number of iterations of the `conjugate_gradient`
    - `P` : pre-computed FFT of ψ_0 needed for the strang_splitting function
    - `P_inv`: pre-computed inverse FFT of ψ_0 needed for the strang_splitting function
    ##### Output values(out): 
    - `out`    : local function """ ->
function choose_integrator(method; tol_cg::Number=1e-10, maxiters_cg::Int=10000, P=nothing, P_inv=nothing)
    methods = ["euler", "crank-nicolson", "strang-splitting"]
    @assert(method in methods, "method = " * string(method) * " must be one of " * string(methods))

    if method == methods[1]
        out = (μ, ϵ, ψ, V, τ) -> euler(μ, ϵ, ψ, V, τ)
    elseif method == methods[2]
        out = (μ, ϵ, ψ, V, τ) -> crank_nicolson(μ, ϵ, ψ, V, τ, tol_cg, maxiters_cg)
    elseif method == methods[3]
        out = (μ, ϵ, ψ, V, τ) -> strang_splitting(μ, ϵ, ψ, V, τ, P, P_inv)
    end
    
    return out
end

@doc """`simulate_1D` creates and saves (to method * ".gif") an animation of the time evolution of a gaussian wavefunction on a chain with N points.
    If observe it also generates and plots the time evolution of the norm of ψ and of the energy, momentum and position expectation values.
    The functions `lattice`, `potential` and `choose_integrator`, `position_expectation`, `momentum_expectation`, `energy_expectation` are being used.
    ##### Input parameters of `simulate_1D`:
    - `N::Int`     : Number of Points on the chain in each direction
    - `μ::Number`  : dimensionless parameter mωr^2/ħ
    - `ϵ::Number`  : dimensionless parameter a/r with a the lattice spacing a
    - `T::Number`  : Total time 
    - `τ::Number`  : time step  
    - `ψ_0::Array` : discretized gaussian wavefunction (q)
    - `method::String`:one of the methods, `euler`, `crank-nicolson` or  `strang-splitting`
    - `fps::Int`   :  frames per second 
    - `gif_length::Number`  : tolerance of the `conjugate_gradient`
    ##### Optional parameters:
    - `observe::Bool`: if true the function also generates plots of the observables
    - `tol_cg::Number`   : tolerance of the `conjugate_gradient`
    - `maxiters_cg::Int` : maximum number of iterations of the `conjugate_gradient`""" ->  
function simulate_1D(N::Int, μ::Number, ϵ::Number, T::Number, τ::Number, ψ_0::Array, method::String, fps::Int, gif_length::Number; observe::Bool=true, tol_cg::Number=1e-10, maxiters_cg::Int=10000)
    @assert(T > 0, "T = " * string(T) * " must be positive")
    @assert(fps > 0, "fps = " * string(fps) * " must be positive")
    @assert(gif_length > 0, "gif_length = " * string(gif_length) * " must be positive")
    
    P = plan_fft(ψ_0); P_inv = inv(P)
    integrator = choose_integrator(method; tol_cg, maxiters_cg, P, P_inv)
    
    n = lattice(N, 1)
    V = potential(μ, ϵ, n)
    ψ = ψ_0; t = 0

    if observe
        times = [0]
        norm_ψ = [norm(ψ)]
        E, E_kin, E_pot = [energy_expectation(ψ, μ, ϵ)[1]], [real(dot(ψ, kinetic(μ, ϵ, ψ)))], [real(dot(ψ, V .* ψ))]
        X, ΔX = [position_expectation(ψ)[1]], [position_expectation(ψ)[2]]
        P, ΔP = [momentum_expectation(ψ)[1]], [momentum_expectation(ψ)[2]]
    end
    
    anim = @animate while t < T
        ψ = integrator(μ, ϵ, ψ, V, τ); t += τ
        fmt = Printf.Format("t = %.1e")
        plot(n*ϵ, abs.(ψ)./sqrt(ϵ), label = Printf.format(fmt, t))
        xlabel!(L"x/r"); ylabel!(L"√r|\psi\,(x)|")
        ylims!(0, 1/sqrt(ϵ))
        plot!(n*ϵ, V/max(V...)./sqrt(ϵ), color = 2, label = "∝ V(n)")
        vline!([-1, 1], color = 2, label = "", linestyle = :dash)

        if observe
            times = vcat(times, [t])
            norm_ψ = vcat(norm_ψ, [norm(ψ)])
            E, E_kin, E_pot = vcat(E, [energy_expectation(ψ, μ, ϵ)[1]]), vcat(E_kin, [real(dot(ψ, kinetic(μ, ϵ, ψ)))]), vcat(E_pot, [real(dot(ψ, V .* ψ))])
            X, ΔX = vcat(X, [position_expectation(ψ)[1]]), vcat(ΔX, [position_expectation(ψ)[2]])
            P, ΔP = vcat(P, [momentum_expectation(ψ)[1]]), vcat(ΔP, [momentum_expectation(ψ)[2]])
        end
    end every round(Int, 1/(gif_length*fps) * T/τ)

    display(gif(anim, method * ".gif", fps = fps))

    if observe
        p_1 = plot(times, norm_ψ/sqrt(ϵ), label = nothing)
        xlabel!(L"t\omega"); ylabel!(L"√r||\psi\,(x)||")
        ylims!(0, max(norm_ψ...)/sqrt(ϵ)*1.1)

        p_2 = plot(times, E, label = "⟨E⟩")
        plot!(times, E_kin, label = "⟨K⟩")
        plot!(times, E_pot, label = "⟨V⟩")
        xlabel!(L"t\omega"); ylabel!(L"E/(\hbar\omega)")

        p_3 = plot(times, map(x -> x[1], X)*ϵ, label = "⟨x⟩")
        plot!(times, map(x -> x[1], ΔX)*ϵ, color = 1, linestyle = :dash, label = "Δx")
        xlabel!(L"t\omega"); ylabel!(L"x/r")

        p_4 = plot(times, map(x -> x[1], P)/ϵ, label = "⟨p⟩")
        plot!(times, map(x -> x[1], ΔP)/ϵ, color = 1, linestyle = :dash, label = "Δp")
        xlabel!(L"t\omega"); ylabel!(L"p/(\hbar/r)")

        p = plot(p_1, p_2, p_3, p_4, layout = (2,2), size = (800,800), left_margin = 1Plots.mm, bottom_margin = 1Plots.mm)
        display(p)
    end
end

@doc """`simulate_2D` creates and saves (to method * "_2D.gif") an animation of the time evolution of a gaussian wavefunction on a two dimensional lattice with N points 
    using the functions `lattice`, `potential` and `choose_integrator`
    ##### Input parameters of `simulate_2D`:
    - `N::Int`     : Number of Points on the lattice in each direction
    - `μ::Number`  : dimensionless parameter mωr^2/ħ
    - `ϵ::Number`  : dimensionless parameter a/r with a the lattice spacing a
    - `T::Number`  : total time 
    - `τ::Number`  : time step  
    - `ψ_0::Array` : discretized gaussian wavefunction (q)
    - `method::String`:one of the methods, `euler`, `crank-nicolson` or  `strang-splitting`
    - `fps::Int`   :  frames per second 
    - `gif_length::Number`  : tolerance of the `conjugate_gradient`
    ##### Optional parameters:
    - `observe::Bool`: if true the function also generates plots of the observables
    - `tol_cg::Number`   : tolerance of the `conjugate_gradient`
    - `maxiters_cg::Int` : maximum number of iterations of the `conjugate_gradient`""" ->
function simulate_2D(N::Int, μ::Number, ϵ::Number, T::Number, τ::Number, ψ_0::Array, method::String, fps::Int, gif_length::Number; observe::Bool=true, tol_cg::Number=1e-10, maxiters_cg::Int=10000)
    @assert(T > 0, "T = " * string(T) * " must be positive")
    @assert(fps > 0, "fps = " * string(fps) * " must be positive")
    @assert(gif_length > 0, "gif_length = " * string(gif_length) * " must be positive")

    P = plan_fft(ψ_0); P_inv = inv(P)
    integrator = choose_integrator(method; tol_cg, maxiters_cg, P, P_inv)
    
    n_1D = lattice(N, 1)
    n = lattice(N, 2)
    V = potential(μ, ϵ, n)
    ψ = ψ_0; t = 0

    if observe
        times = [0]
        norm_ψ = [norm(ψ)]
        E, E_kin, E_pot = [energy_expectation(ψ, μ, ϵ)[1]], [real(dot(ψ, kinetic(μ, ϵ, ψ)))], [real(dot(ψ, V .* ψ))]
        X, ΔX = [position_expectation(ψ)[1]], [position_expectation(ψ)[2]]
        P, ΔP = [momentum_expectation(ψ)[1]], [momentum_expectation(ψ)[2]]
    end
    
    anim = @animate while t < T
        ψ = integrator(μ, ϵ, ψ, V, τ); t += τ
        fmt = Printf.Format("t = %.1e")
        surface(n_1D*ϵ, n_1D*ϵ, abs.(ψ)./ϵ, cmap = :blues, title = Printf.format(fmt, t), legend = nothing)
        xlabel!(L"x/r"); ylabel!(L"y/r"); zlabel!(L"r|\psi\,(\textbf{x})|")
        zlims!(0, 0.1/ϵ)
        #surface!(n_1D, n_1D, V/max(V...) * 0.1, cmap = :heat, alpha = :0.3, legend = nothing)
        #plot sircle of minima in the x-y-plane?

        if observe
            times = vcat(times, [t])
            norm_ψ = vcat(norm_ψ, [norm(ψ)])
            E, E_kin, E_pot = vcat(E, [energy_expectation(ψ, μ, ϵ)[1]]), vcat(E_kin, [real(dot(ψ, kinetic(μ, ϵ, ψ)))]), vcat(E_pot, [real(dot(ψ, V .* ψ))])
            X, ΔX = vcat(X, [position_expectation(ψ)[1]]), vcat(ΔX, [position_expectation(ψ)[2]])
            P, ΔP = vcat(P, [momentum_expectation(ψ)[1]]), vcat(ΔP, [momentum_expectation(ψ)[2]])
        end
    end every round(Int, 1/(gif_length*fps) * T/τ)

    display(gif(anim, method * "_2D.gif", fps = fps))

    if observe
        p_1 = plot(times, norm_ψ/ϵ, label = nothing)
        xlabel!(L"t\omega"); ylabel!(L"r||\psi\,(\textbf{x})||")
        ylims!(0, max(norm_ψ...)/ϵ*1.1)

        p_2 = plot(times, E, label = "⟨E⟩")
        plot!(times, E_kin, label = "⟨K⟩")
        plot!(times, E_pot, label = "⟨V⟩")
        xlabel!(L"t\omega"); ylabel!(L"E/(\hbar\omega)")

        p_3 = plot(times, map(x -> x[1], X)*ϵ, label = "⟨x⟩")
        plot!(times, map(x -> x[1], ΔX)*ϵ, color = 1, linestyle = :dash, label = "Δx")
        plot!(times, map(x -> x[2], X)*ϵ, color = 2, label = "⟨y⟩")
        plot!(times, map(x -> x[2], ΔX)*ϵ, color = 2, linestyle = :dash, label = "Δy")
        xlabel!(L"t\omega"); ylabel!(L"x/r")

        p_4 = plot(times, map(x -> x[1], P)/ϵ, label = "⟨p_x⟩")
        plot!(times, map(x -> x[1], ΔP)/ϵ, color = 1, linestyle = :dash, label = "Δp_x")
        plot!(times, map(x -> x[2], P)/ϵ, color = 2, label = "⟨p_y⟩")
        plot!(times, map(x -> x[2], ΔP)/ϵ, color = 2, linestyle = :dash, label = "Δp_y")
        xlabel!(L"t\omega"); ylabel!(L"p/(\hbar/r)")

        p = plot(p_1, p_2, p_3, p_4, layout = (2,2), size = (800,800), left_margin = 1Plots.mm, bottom_margin = 1Plots.mm)
        display(p)
    end
end