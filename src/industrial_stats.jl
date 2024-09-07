module IndustrialStats

include("cexch/cexch.jl")
include("model/model_builder.jl")
include("model/design_initializer.jl")
include("optim/design_optimizer.jl")
include("optim/optimality_criteria.jl")
include("util/util.jl")

using .CEXCH
using .DesignInitializer
using .ModelBuilder
using .DesignOptimizer
using .OptimalityCriteria
using .Util

end