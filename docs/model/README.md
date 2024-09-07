# ModelBuilder: A Julia Module for Constructing Model Matrices from Design Matrices
ModelBuilder is a Julia module designed to streamline the creation of model matrices from experimental designs. 

## Usage
ModelBuilder was designed to operate on design tensors of shape $n\times N\times K$, where $n$ is the number of designs being considered, $N$ is the number of design points, and $K$ is the number of experimental factors.

Both the `create` family of functions and all the default models (e.g. `ModelBuilder.linear`) return model builders, which are functions $F$ with

$$ F: \mathbb{R}^{n\times N\times K} \rightarrow \mathbb{R}^{n\times N\times p} $$

where $p$ is the number of parameters in the model.

Model builders can be re-used across different designs and applied to one-, two-, and three-dimensional Julia arrays.

### Basic Usage
```julia
import ModelBuilder

# Get linear model builder
model_builder = ModelBuilder.linear

# Build model matrices for 1000 experimental designs on K=5 factors and N=12 design points
X = rand(1000, 12, 5)
F = model_builder(X) # 1000x12xp tensor

# Single design matrix for K=5, N=12
X = rand(12, 5)
F = model_builder(X) # 12xp matrix

# Single design point for K=5
X = rand(5)
F = model_builder(X) # vector length p

# The factory provides many useful parameters for customizing model creation
# Produces a model with an intercept, first/second order power terms, and second order interactions 
model_builder = ModelBuilder.create(intercept=true, order=2, interaction_order=2)
```

### Advanced Usage

#### Example 1
$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \beta_4 x_1x_2 + \beta_5 x_2x_3 + \beta_6 x_1x_3 + \beta_7 x_1^2 + \beta_8 x_2^2 + \beta_9 x_3^2
$$


```julia
### Method One: Built-In Models (Complete list can be found below)
model_builder = ModelBuilder.quadratic_interaction

### Method Two: Default factory parameters
model_builder = ModelBuilder.create(order=2, interaction_order=1)

### Method Three: Imperative Method
model_builder = ModelBuilder.builder()
model_builder = model_builder.add(intercept())
model_builder = model_builder.add(powers([1, 2, 3], [1, 2])) # First and second order terms for all three factors
model_builder = model_builder.add(interactions([1, 2, 3], [1])) # First order interactions for all three factors

### Method Four: Vector of functions
funcs = [
    intercept(), # Intercept
    powers([1, 2, 3], [1, 2]), # First and second order terms for all three factors
    interactions([1, 2, 3], [1]), # First order interactions for all three factors
]

# Create model builder
model_builder = ModelBuilder.create(funcs)

# Apply to tensor of 1000 experimental designs on K=5 factors and N=12 design points
X = rand(1000, 12, 5)
F = model_builder(X)
```

#### Example 2
$$
    y = \beta_0 + \beta_1x_1 + \beta_2\exp( x_2) + \beta_3 x_3 + \beta_4x_1x_3+\beta_5x_1^2+\beta_6x_2^2+\beta_7x_3^3+\beta_8 x_1x_2x_3
$$

```julia
# Define vector of functions to apply to design matrix
funcs = [
    intercept(),
    powers([1, 3], [1, 2]), # First and second order terms for the first and third factor
    powers([3], [3]), # Third order term for third factor
    interactions([1, 3], [1]), # First order interaction for first and third factor
    x -> exp.(x(2)), # x is a function of the factor index, so x(2) represents the second factor
    x -> x(1) .* x(2) .* x(3) # Another way to define an interaction term as a custom function
]

# Create model builder
model_builder = ModelBuilder.create(funcs)

# Apply to tensor of 1000 experimental designs on K=5 factors and N=12 design points
X = rand(1000, 12, 5)
F = model_builder(X)
```

## Default Models

- `linear`: Basic linear model.
- `quadratic`: Quadratic model with first- and second-order power terms.
- `cubic`: Cubic model with first-, second-, and third-order power terms.
- `linear_interaction`: Linear model with second-order interaction terms.
- `quadratic_interaction`: Quadratic model with first- and second-order power terms and second-order interaction terms.
- `cubic_interaction`: Cubic model with first-, second-, and third-order power terms and second-order interaction terms.
- `scheffe(n)`: Scheffe model for mixture designs. Includes first order terms and $n-1$-th order interaction terms, with no intercept.  

## Available Functions
- `create(; order=1, interaction_order=1, include_intercept=true, transforms=[])`: Constructs a model builder with specified parameters.
- `create(funcs::Vector)`: Creates a model builder for the vector of functions defining the model.
- `intercept()`: Generates an intercept term for the model tensor.
- `powers(factors, powers)`: Generates power terms for model building. Factors is a vector of column indices corresponding with experimental factors; powers are the exponents to which each factor will be raised.
- `interactions(factors, orders)`: Creates interaction terms for model building. Factors is a vector of column indices; orders is a vector of orders. All combinations of factors of length order for each order in the orders vector will be constructed.
- `add`: Adds a transform function to the model builder. The transform function is applied to the model tensor.
- `expand(X; left=true)`: Reshapes the input to a 3D tensor.
- `squeeze(X)`: Reshapes the input to remove singleton dimensions.


## Conclusion

ModelBuilder offers a convenient and efficient way to construct and customize model matrices for experimental designs in Julia, leveraging functional programming paradigms for flexibility and performance.
