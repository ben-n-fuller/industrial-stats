module IndustrialStats

include("cexch/cexch.jl")
include("model/model_builder.jl")
include("model/design_initializer.jl")
include("optim/design_optimizer.jl")
include("optim/optimality_criteria.jl")
include("util/tensor_ops.jl")
include("util/distributed_jobs.jl")
include("util/aitchison_geom.jl")

using .DistributedJobs
using .TensorOps
using .CEXCH
using .DesignInitializer
using .ModelBuilder
using .DesignOptimizer
using .OptimalityCriteria
using .AitchisonGeometry

end