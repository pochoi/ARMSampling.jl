module ARMSampling

import Base: length, copy, insert!
export TrapezoidalProposal, sample, sample!


#From StatsBase.jl/src/sampling.jl
function sample(w)
    t = rand() * sum(w)
    n = length(w)
    i = 1
    cw = w[1]
    while cw < t && i < n
        i += 1
        @inbounds cw += w[i]
    end
    return i
end

function rank(out)
    x0 = out[1]
    sorted_out = sort(out)
    index = searchsorted(sorted_out, x0)

    if index.start == index.stop
        r = index.start - 1
    else
        r = sample(index) - 1
    end
    return r, sorted_out
end

slope(x0, x1, y0, y1) = (y1-y0)/(x1-x0)
intercept(x0, x1, y0, y1) = (x1*y0 - x0*y1)/(x1-x0)
slope_intercept(x0, x1, y0, y1) = (y1-y0)/(x1-x0), (x1*y0 - x0*y1)/(x1-x0)
straight_line_fun(x, x0, x1, y0, y1) = slope(x0, x1, y0, y1)*x + intercept(x0, x1, y0, y1)

function integrate_straight_line_exp(x0, x1, y0, y1)
    a, b = slope_intercept(x0, x1, y0, y1)
    if abs(a) < eps(typeof(a))
        return exp(b)*(x1-x0)
    end

    return (exp(a*x1+b)-exp(a*x0+b))/a
end

function trapezoidalweight(x::AbstractVector{T}, y::AbstractVector{T}) where T
    n = length(x) - 2
    w = T[integrate_straight_line_exp(x[i], x[i+1], y[i], y[i+1]) for i in 1:(n+1)]
    return w
end

struct TrapezoidalProposal{T,V<:AbstractVector{T}}
    x::V
    y::V
    w::V

    function TrapezoidalProposal{T,V}(x::V, y::V, w::V) where {T,V<:AbstractVector{T}}
        @assert issorted(x, lt = (x,y) -> x <= y)
        @assert length(x) >= 3
        @assert length(y) == length(x)
        @assert length(w) == length(x) - 1
        new{T,V}(x, y, w)
    end
end

function TrapezoidalProposal(x::AbstractVector{T}, y::AbstractVector{T}) where T
    w = trapezoidalweight(x, y)
    return TrapezoidalProposal{T,typeof(x)}(x, y, w)
end

function TrapezoidalProposal(x::AbstractVector{T}, logf::Function) where T
    y = T[logf(xx) for xx in x]
    return TrapezoidalProposal(x, y)
end

length(proposal::TrapezoidalProposal) = length(proposal.x) - 2

function copy(proposal::TrapezoidalProposal{T, V}) where {T,V<:AbstractVector{T}}
    newx = copy(proposal.x)
    newy = copy(proposal.y)
    neww = copy(proposal.w)
    TrapezoidalProposal{T,V}(newx, newy, neww)
end

function insert!(proposal::TrapezoidalProposal{T}, x0::T, y0::T) where T
    n = length(proposal)
    x = proposal.x
    y = proposal.y
    w = proposal.w
    
    index = searchsorted(x, x0)
    left = index.stop
    right = index.start

    if left == right
        return proposal
    elseif left == 0
        w0 = integrate_straight_line_exp(x0, x[1], y0, y[1])
        insert!(w, 1, w0)
    elseif right == n + 3
        w0 = integrate_straight_line_exp(x[n+2], x0, y[n+2], y0)
        insert!(w, n+2, w0)
    else
        w_left  = integrate_straight_line_exp(x[left], x0, y[left], y0)
        w_right = integrate_straight_line_exp(x0, x[right], y0, y[right])
        splice!(w, left, [w_left, w_right])
    end
    insert!(x, right, x0)
    insert!(y, right, y0)
    return proposal
end

insert!(proposal::TrapezoidalProposal{T}, x0::T, logf::Function) where T = insert!(proposal, x0, logf(x0))

function (proposal::TrapezoidalProposal)(x0)
    if x0 < proposal.x[1] || x0 > proposal.x[end]
        error("x out of range of proposal: x = $x0")
    end
    x = proposal.x
    y = proposal.y
    index = searchsorted(x, x0)
    left = index.stop
    right = index.start
    if left == right
        return y[left]
    end

    return straight_line_fun(x0, x[left], x[right], y[left], y[right])
end

function sample(proposal::TrapezoidalProposal)
    x = proposal.x
    y = proposal.y
    w = proposal.w

    n = length(proposal)
    k = sample(w)

    a, b = slope_intercept(x[k], x[k+1], y[k], y[k+1])
    if abs(a) < eps(typeof(a))
        return rand()*w[k]*exp(-b) + x[k]
    end

    out = log(rand()*w[k]*a*exp(-b)+exp(a*x[k]))/a
    if !isfinite(out)
        println()
        error("Got infinity! k = $k, a = $a, b = $b")
    end
    return out
end

function sample!(proposal::TrapezoidalProposal{T}, logf::Function, x0::T; max_n = 2*length(proposal)) where T
    logfx0 = logf(x0)
    logfx = zero(T)
    x = zero(T)
    while true
        x = sample(proposal)
        logfx = logf(x)
        if rand() > exp(logfx - proposal(x))
            if length(proposal) < max_n
                insert!(proposal, x, logfx)
            end
        else
            break
        end
    end

    if rand() > min(1, exp(logfx - logfx0)*min(exp(logfx0), exp(proposal(x0)))/min(exp(logfx),exp(proposal(x))) )
    #if rand() > min(1, exp(logfx - logfx0 + proposal(x0) - proposal(x)))

        if length(proposal) < max_n
            insert!(proposal, x0, logfx0)
        end

        return x0
    else
        return x
    end
end

function sample!(out::Vector{T}, nsample::Int, proposal::TrapezoidalProposal{T}, logf::Function; kwargs...) where T
    nout = length(out)
    append!(out, zeros(T, nsample))
    for i in 1:nsample
        out[i+nout] = sample!(proposal, logf, out[i+nout-1]; kwargs...)
    end
    return out
end


function overrelaxation!(proposal::TrapezoidalProposal{T}, logf::Function, x0::T, K::Int; max_n = 2*length(proposal), max_try = 10*length(proposal)) where T

    x = x0
    max_try_counter = 0
    while max_try_counter < max_try && length(proposal) < max_n
        x = sample!(proposal, logf, x; max_n = max_n)
        max_try_counter += 1
    end

    out = T[x0]
    sample!(out, 2*K, proposal, logf; max_n = max_n)
    out = out[1:2:end]
    r, sorted_out = rank(out)

    return sorted_out[K-r+1]
end

end # module
