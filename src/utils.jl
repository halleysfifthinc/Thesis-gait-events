using DatasetManager, Statistics, Peaks
using C3D: C3DFile
using DSP: Butterworth, Lowpass, digitalfilter, filtfilt
using NearestNeighbors: KDTree, nn
using Distances: Chebyshev
using Biomechanics: centraldiff, ForwardBackwardPad, totimes

const df = let
    Fc = 12
    n = 2
    bw = Butterworth(n)
    corrfac = inv((2^inv(n)-1)^(1/4)) # Correction factor for Fc of multi-pass filters
    lpf = Lowpass(Fc * corrfac; fs=100)
    digitalfilter(lpf, bw)
end

function getheel(trial::Trial; kwargs...)
    getheel(readsource(getsource(trial, Source{C3DFile})); kwargs...)
end

function getheel(file::C3DFile;
    lheelmkr="LHEE",
    rheelmkr="RHEE",
    axis=Colon(),
    start=firstindex(file.point[lheelmkr],1),
    finish=lastindex(file.point[lheelmkr],1)
)
    lheel = file.point[lheelmkr]
    rheel = file.point[rheelmkr]

    frheel = filtfilt(df, rheel[start:finish,axis])
    flheel = filtfilt(df, lheel[start:finish,axis])

    return flheel, frheel
end

function roerdink2008(trial; fo_minprom=1200)
    c3dsrc = readsource(getsource(trial, Source{C3DFile}); strip_prefixes=true)
    fs = c3dsrc.groups[:POINT][Int, :RATE]

    lheel, rheel = getheel(c3dsrc; axis=3)
    lfcpred, _ = peakproms!(argminima(lheel, 10), lheel; minprom=30)
    rfcpred, _ = peakproms!(argminima(rheel, 10), rheel; minprom=30)

    lheel_vel = centraldiff(lheel; dt=inv(fs), padding=ForwardBackwardPad())
    rheel_vel = centraldiff(rheel; dt=inv(fs), padding=ForwardBackwardPad())

    lfopred, _ = peakproms!(argmaxima(lheel_vel, 10), lheel_vel; minprom=fo_minprom)
    rfopred, _ = peakproms!(argmaxima(rheel_vel, 10), rheel_vel; minprom=fo_minprom)

    return Dict("LFC" => totimes(lfcpred, fs), "RFC" => totimes(rfcpred, fs),
                "LFO" => totimes(lfopred, fs), "RFO" => totimes(rfopred, fs))
end

"""
    findduplicates(itr)

"""
function findduplicates(itr)
    dups = Dict{eltype(itr), Vector{Int}}()

    for (i, v) in enumerate(itr)
        push!(get!(dups, v, Int[]), i)
    end
    filter!(((k,v),) -> length(v) !== 1, dups)

    return dups
end

function matchevents(pred::AbstractVector, act::AbstractVector; Tthresh=0.5*median(diff(act)))
    mdiff_pred = median(diff(pred))
    mdiff_act = median(diff(act))
    err(a,b) = abs(a - b)/b
    if err(mdiff_pred, mdiff_act) > 10 || err(mdiff_act, mdiff_pred) > 10
        throw(error("""detected a large difference in event frequency. Are predicted events
            in the same units as actual events (eg frames vs sec)?"""))
    end
    missed = max(0, length(pred) - length(act))

    _pred = copy(pred)
    tree = KDTree(reshape(act, 1, length(act)), Chebyshev())
    idxs, dists = nn(tree, reshape(pred, 1, length(pred)))

    dups = findduplicates(idxs)
    delidxs = Int[]
    if !isempty(dups)
        foreach(dups) do (k, v)
            append!(delidxs, v[setdiff(eachindex(v), argmin(dists[v]))])
        end
        sort!(delidxs)
        deleteat!(idxs, delidxs)
        deleteat!(dists, delidxs)
        deleteat!(_pred, delidxs)
    end

    dists .= _pred - act[idxs]

    return setdiff(eachindex(pred), delidxs), idxs, dists, missed
end

