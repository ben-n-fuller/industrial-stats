@testset "expand function tests" begin
    # Test adding singleton dimensions to the left
    @testset "Adding singleton dimensions to the left" begin
        X = [1, 2, 3]
        Y = ModelBuilder.expand(X)
        @test ndims(Y) == 3
        @test size(Y) == (1, 1, 3)
    end

    # Test adding singleton dimensions to the right
    @testset "Adding singleton dimensions to the right" begin
        X = [1, 2, 3]
        Y = ModelBuilder.expand(X, left=false)
        @test ndims(Y) == 3
        @test size(Y) == (3, 1, 1)
    end

    # Test with the number of dimensions already equal to `n`
    @testset "Input with desired number of dimensions" begin
        X = reshape([1, 2, 3, 4], (1, 2, 2))
        Y = ModelBuilder.expand(X)
        @test Y === X  # Using `===` to test for exact equality (same object)
    end

    # Test with different array types
    @testset "Different array types" begin
        X = reshape([1, 2, 3, 4], (2, 2))  # Matrix
        Y = ModelBuilder.expand(X)
        @test ndims(Y) == 3
        @test size(Y) == (1, 2, 2)
    end
end

@testset "squeeze function tests" begin
    # Test removing singleton dimensions
    @testset "Removing singleton dimensions" begin
        X = reshape([1, 2, 3, 4], (1, 2, 2))
        Y = ModelBuilder.squeeze(X)
        @test ndims(Y) == 2
        @test size(Y) == (2, 2)
    end

    # Test with no singleton dimensions
    @testset "Input with no singleton dimensions" begin
        X = [1, 2, 3]
        Y = ModelBuilder.squeeze(X)
        @test Y === X  # Using `===` to test for exact equality (same object)
    end
end

