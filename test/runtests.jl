using Test
using ARMSampling
using Statistics

@testset "TrapezoidalProposal" begin
    logf(x) = -(x-1)^2
    x = [-3.0, 0.0, 1.0, 2.0, 5.0]
    y = [logf(x1) for x1 in x]

    proposal = TrapezoidalProposal(x, y)
    @test length(proposal) == 3
    @test proposal.w[2] ≈ (ℯ - 1.0)/ℯ
    @test proposal.w[3] ≈ (ℯ - 1.0)/ℯ
    @test x[1] <= sample(proposal) <= x[end]

    insert!(proposal, 0.5, logf(0.5))
    @test length(proposal) == 4
    @test sum(proposal.w[2:3]) > proposal.w[4]

    insert!(proposal, 1.5, logf(1.5))
    @test length(proposal) == 5
    @test sum(proposal.w[2:3]) ≈ sum(proposal.w[4:5])
    @test x[1] <= sample!(proposal, logf, 1.1) <= x[end]

    mysample = [1.1]
    sample_size = 20000
    sample!(mysample, sample_size, proposal, logf)
    @test length(mysample) == sample_size + 1
    @test all( proposal.x[1] .<= mysample .<= proposal.x[end] )
    @test abs(std(mysample) - sqrt(0.5)) < 1e-2
    @test abs(mean(mysample) - 1.0) < 1e-2
end
