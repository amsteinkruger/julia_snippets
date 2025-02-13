This document compiles code used in open courses at https://juliaacademy.com/.

# Introduction to Julia
## Preliminaries, Data Types, and Strings


```julia
println("I'm learning Julia!")

my_answer = 42
typeof(my_answer)

my_pi = 3.14159
typeof(my_pi)

😺 = "smiley cat!"
typeof(😺)

😺 = 1
typeof(😺)

😀 = 0
😞 = -1

😺 + 😞 == 😀

# You can leave comments on a single line using the pound/hash key

#=

For multi-line comments, 
use the '#= =#' sequence.

=#

#=

This is another example.

=#

sum = 3 + 7
difference = 10 - 3
product = 20 * 5
quotient = 100 / 10
power = 10 ^ 2
modulus = 101 % 2

days = 365
days_float = convert(Float16, days)
typeof(days_float)

@assert days == 365
@assert days_float == 365.0

s1 = "I am a string."
s2 = """I am also a string."""

# "Here, we get an "error" because it's ambiguous where this string ends"

"""Look, Mom, no "errors"!!! """

# Note that ' ' define a character, but NOT a string!
typeof('a')

name = "Jane"
num_fingers = 10
num_toes = 10
println("Hello, my name is $name.")
println("I have $num_fingers fingers and $num_toes toes.")
println("That is $(num_fingers + num_toes) digits in all!!")

s3 = "How many cats ";
s4 = "is too many cats?";
😺 = 10

string(s3, s4)
string("I don't know, but ", 😺, " is too few.")
s3*s4

repeat("hi", 3) == "hi"^3
hi = repeat("hi", 3)
@assert hi == "hihihi"

a = 3
b = 4
c = string(a, " + ", b)
d = string(a + b)
@assert c == "3 + 4"
@assert d == "7"
```

## Data Structures


```julia
# Tuples

myfavoriteanimals = ("penguins", "cats", "sugargliders") # is a tuple
myfavoriteanimals[1]
# since tuples are immutable, we can't update it
myfavoriteanimals = (bird = "penguins", mammal = "cats", marsupial = "sugargliders")
myfavoriteanimals[1]
myfavoriteanimals.bird

# Dictionaries

myphonebook = Dict("Jenny" => "867-5309", "Ghostbusters" => "555-2368")
myphonebook["Jenny"]
myphonebook["Kramer"] = "555-FILK"
pop!(myphonebook, "Kramer")

# Arrays

myfriends = ["Ted", "Robyn", "Barney", "Lily", "Marshall"]
fibonacci = [1, 1, 2, 3, 5, 8, 13]
mixture = [1, 1, 2, 3, "Ted", "Robyn"]
myfriends[3]
myfriends[3] = "Baby Bop"
push!(fibonacci, 21)
pop!(fibonacci)
favorites = [["koobideh", "chocolate", "eggs"],["penguins", "cats", "sugargliders"]]
numbers = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
rand(4, 3)
rand(4, 3, 2)
fibonacci
somenumbers = fibonacci
somenumbers[1] = 404
fibonacci
# Note that somenumbers is an alias for fibonacci, not a distinct object. copy does what it says on the tin.
fibonacci[1] = 1
fibonacci
somemorenumbers = copy(fibonacci)
somemorenumbers[1] = 404
fibonacci
```

## Loops


```julia
n = 0
while n < 10
    n += 1
    println(n)
end
n

myfriends = ["Ted", "Robyn", "Barney", "Lily", "Marshall"]

i = 1
while i <= length(myfriends)
    friend = myfriends[i]
    println("Hi $friend, it's great to see you!")
    i += 1
end

for n in 1:10
    println(n)
end

myfriends = ["Ted", "Robyn", "Barney", "Lily", "Marshall"]

for friend in myfriends
    println("Hi $friend, it's great to see you!")
end

m, n = 5, 5
A = fill(0, (m, n))
for j in 1:n
    for i in 1:m
        A[i, j] = i + j
    end
end
A

B = fill(0, (m, n))
for j in 1:n, i in 1:m
    B[i, j] = i + j
end
B

C = [i + j for i in 1:m, j in 1:n] # This is an array comprehension and thus stylish?

# Exercises

# (1)

for i in 1:10
    println(i^2)
end

# (2)

n = 3
squares = Dict()

for i in 1:n
    squares[(i)] = i^2
end

@assert squares[2] == 4
@assert squares[3] == 9

# (3)

# Get a matrix where squares are on the diagonal. This isn't what's asked for, but it's neat.
squares_array = [i * j for i in 1:n, j in 1:n]

# Get a vector instead.
squares_array = [i^2 for i in 1:n]

@assert length(squares_arr) == n
@assert sum(squares_arr) == 14
```

## Conditionals


```julia
# Ex. 1

N = 3

if (N % 3 == 0) && (N % 5 == 0) # `&&` means "AND"; % computes the remainder after division
    println("FizzBuzz")
elseif N % 3 == 0
    println("Fizz")
elseif N % 5 == 0
    println("Buzz")
else
    println(N)
end

# Ex. 2

x = 2
y = 3

if x > y
    x
else
    y
end

(x > y) ? x : y

# Ex. 3

false && (println("hi"); true)
true && (println("hi"); true)
# (x > 0) && error("x cannot be greater than 0")
true || println("hi")
false || println("hi")

# Exercises

x = 3

if x % 2 == 0
    println(x)
else
    println("odd")
end

(x % 2 == 0) ? x : "odd"
```

## Functions


```julia
# Three ways to declare functions

function sayhi(name)
    println("Hi $name, it's great to see you!")
end

function f(x)
    x^2
end

sayhi("C-3PO")
f(42)

sayhi2(name) = println("Hi $name, it's great to see you!")
f2(x) = x^2
sayhi2("R2D2")
f2(42)

sayhi3 = name -> println("Hi $name, it's great to see you!")
f3 = x -> x^2
sayhi3("Chewbacca")
f3(42)

# Duck-Typing

sayhi(55595472)

A = rand(3, 3)
A

f(A)
f("hi")
v = rand(3)
# This won't work.
# f(v) 

# (Non-)Mutating Functions

# Note !.

v = [3, 5, 2]

sort(v)
v

sort!(v)
v

# Higher-Order Functions

map(f, [1, 2, 3])

x -> x^3
map(x -> x^3, [1, 2, 3])

broadcast(f, [1, 2, 3])
f.([1, 2, 3])

A = [i + 3*j for j in 0:2, i in 1:3]
f(A)
B = f.(A)
A .+ 2 .* f.(A) ./ A

# instead of 

broadcast(x -> x + 2 * f(x) / x, A)

# Exercises

# (1)

add_one(input) = input + 1
add_one(3)

@assert add_one(1) == 2
@assert add_one(11) == 12

# (2)

A = [i + 3*j for j in 0:2, i in 1:3]
A

A1 = add_one.(A)
A1

@assert A1 == [2 3 4; 5 6 7; 8 9 10]

# (3)

A2 = add_one.(A1)

@assert A2 == [3 4 5; 6 7 8; 9 10 11]
```

## Packages


```julia
using Pkg
Pkg.add("Example")
using Example

hello("it's me. I was wondering if after all these years you'd like to meet.")

Pkg.add("Colors")
using Colors
palette = distinguishable_colors(100)
rand(palette, 3, 3)

# Exercises

# (1)

using Pkg
Pkg.add("Primes")
using Primes

@assert @isdefined Primes

# (2)

primes_list = primes(0, 100)

@assert primes_list == primes(100)
```

## Plots


```julia
using Pkg
Pkg.add("Plots")
using Plots

globaltemperatures = [14.4, 14.5, 14.8, 15.2, 15.5, 15.8]
numpirates = [45000, 20000, 15000, 5000, 400, 17];
gr()

plot(numpirates, globaltemperatures, label="line")  
scatter!(numpirates, globaltemperatures, label="points")
xlabel!("Number of Pirates [Approximate]")
ylabel!("Global Temperature (C)")
title!("Influence of pirate population on global warming")

xflip!()

Pkg.add("UnicodePlots")
unicodeplots()

plot(numpirates, globaltemperatures, label="line")  
scatter!(numpirates, globaltemperatures, label="points") 
xlabel!("Number of Pirates [Approximate]")
ylabel!("Global Temperature (C)")
title!("Influence of pirate population on global warming")

# Exercises

# (1)
# (2)
p1 = plot(x, x)
p2 = plot(x, x.^2)
p3 = plot(x, x.^3)
p4 = plot(x, x.^4)
plot(p1, p2, p3, p4, layout = (2, 2), legend = false)
```

## Multiple Dispatch


```julia
f(x) = x.^2
f(10)
f([1, 2, 3])

# Specifying Methods

foo(x::String, y::String) = println("My inputs x and y are both strings!")
foo("hello", "hi!")
# foo(3, 4)
foo(x::Int, y::Int) = println("My inputs x and y are both integers!")
foo(3, 4)
foo("hello", "hi!")
methods(foo)
methods(+)
@which foo(3, 4)
@which 3.0 + 3.0
foo(x::Number, y::Number) = println("My inputs x and y are both numbers!")
foo(3.0, 4.0)
foo(x, y) = println("I accept inputs of any type!")
v = rand(3)
foo(v, v)

# Exercises

# (1)

# (2)

foo(true)

@assert foo(true) == "foo with one boolean!"



```

## Speed


```julia
a = rand(10^7)
sum(a)
@time sum(a)

using Pkg
Pkg.add("BenchmarkTools")
using BenchmarkTools  

# Note subsequent material on benchmarking with C., etc.
```

## Linear Algebra


```julia
# Note full material defines each operation to avoid ambiguous interpretation. Worth reading. 

A = rand(1:4,3,3)

x = fill(1.0, (3,))

b = A*x

A'

transpose(A)

A'A

A\b

Atall = rand(3, 2)

Atall\b

v = rand(3)
rankdef = hcat(v, v)
rankdef\b

bshort = rand(2)
Ashort = rand(2, 3)

Ashort\bshort

# Exercises

# (1)

using LinearAlgebra

v = [1,2,3]

v_dot = v' * v

@assert v_dot == 14

# (2)

v_outer = v * v'

@assert v_outer == [1 2 3
                    2 4 6
                    3 6 9]

# (3)

v_cross = cross(v, v)

@assert v_cross == [0, 0, 0]

```

## Factorizations


```julia
# LU

using LinearAlgebra
A = rand(3, 3)
x = fill(1, (3,))
b = A * x

Alu = lu(A)
typeof(Alu)
Alu.P
Alu.L
Alu.U

A\b
Alu\b

det(A) ≈ det(Alu)

# QR

Aqr = qr(A)
Aqr.Q
Aqr.R

# Eigendecompositions

Asym = A + A'
AsymEig = eigen(Asym)
AsymEig.values
AsymEig.vectors
inv(AsymEig)*Asym

# Special Matrix Structures

n = 1000
A = randn(n,n);

Asym = A + A'
issymmetric(Asym)

Asym_noisy = copy(Asym)
Asym_noisy[1,2] += 5eps()

issymmetric(Asym_noisy)

Asym_explicit = Symmetric(Asym_noisy);

# Note benchmarking interlude.

# Note "generic" (?) linear algebra section.

# Note exercises.
```

# Data Science with Julia

## Dealing with Data


```julia
# Get packages.

using BenchmarkTools
using DataFrames
using DelimitedFiles
using CSV
using XLSX
using Downloads

# Use Downloads functions ("download") for files online.

# dat_download = Downloads.download("https://raw.githubusercontent.com/nassarhuda/easy_data/master/programming_languages.csv", "programming_languages.csv")

# Use readdlm for simple delimited files.

dat_readdlm = readdlm("data/programming_languages.csv", ','; header = true)

# Use CSV functions ("read") for less simple delimited files.

dat_read = CSV.read("data/programming_languages.csv", DataFrame)

# Use DelimitedFiles functions for complex delimited files.

# Look at objects.
@show typeof(dat_readdlm)
@show typeof(dat_read)

names(dat_read)
describe(dat_read)

# Write an object to disk.

CSV.write("out/programming_languages_out.csv", dat_read)
XLSX.writetable("out/programming_languages_out.xlsx", dat_read, overwrite = true)

# Use XLSX functions ("readdata") for spreadsheets. 
dat_zillow = XLSX.readdata("data/zillow.xlsx", # file name
    "Sale_counts_city", # sheet name
    "A1:F9" # cell range
    )

# Untrue: The resulting object is a tuple of (1) a vector of vectors, with each vector a column and (2) column names.
# True: the resulting object is a matrix, because something about XLSX changed since the Data Science course was developed.

@show typeof(dat_zillow)
dat_zillow

# Writing that to a dataframe might be nice.
#  This is another case of the course's code not working after updates.
#  Cheers to contributors at https://stackoverflow.com/questions/26769162/convert-julia-array-to-dataframe.

dat_zillow_dataframe = DataFrame(dat_zillow[2:end, 1:end], string.(dat_zillow[1, 1:end]))

# Writing a dataframe from scratch might be nice, too.

foods = ["apple", "cucumber", "tomato", "banana"]
calories = [105, 47, 22, 105]
prices = [0.85, 1.6, 0.8, 0.6,]
dat_calories = DataFrame(item = foods, calories = calories)
dat_prices = DataFrame(item = foods, price = prices)

# Joins are a thing.

dat_join = innerjoin(dat_calories, dat_prices, on=:item)

# Note course section on reading files of different types, including .jld, .npz, .rda, and .mat. 

# On to asking initial questions about the programming languages dataset.

# Get the year a language was invented.

function year_created(dat_read ,language::String)
    loc = findfirst(dat_read[:,2] .== language)
    return dat_read[loc,1]
end

year_created(dat_read,"Julia")

# What if the user asks about a missing language?

function year_created_handle_error(dat_read, language::String)
    loc = findfirst(dat_read[:, 2] .== language)
    !isnothing(loc) && return dat_read[loc, 1]
    error("Error: Language not found.")
end

# year_created_handle_error(dat_read, "W")

# Count languages invented in a year.

function how_many_per_year(dat_read, year::Int64)
    year_count = length(findall(dat_read[:,1].==year))
    return year_count
end

how_many_per_year(dat_read, 2011)

# The dataset might be easier to deal with in a dataframe.

dat_read_dataframe = DataFrame(dat_read)

# Then the same functions will look a little different:

function year_created(dat_read_dataframe, language::String)
    loc = findfirst(dat_read_dataframe.language .== language)
    return dat_read_dataframe.year[loc]
end

year_created(dat_read_dataframe, "Julia")

function year_created_handle_error(dat_read_dataframe, language::String)
    loc = findfirst(dat_read_dataframe.language .== language)
    !isnothing(loc) && return dat_read_dataframe.year[loc]
    error("Error: Language not found.")
end

# year_created_handle_error(dat_read_dataframe, "W")

function how_many_per_year(dat_read_dataframe, year::Int64)
    year_count = length(findall(dat_read_dataframe.year.==year))
    return year_count
end

how_many_per_year(dat_read_dataframe, 2011)

# Dictionaries are another type for data.

Dict([("A", 1), ("B", 2), (1, [1, 2])])

dat_dict = Dict{Integer, Vector{String}}()

# Check out the inline note.

for i = 1:size(dat_read, 1)
    year,lang = dat_read[i,:]
    if year in keys(dat_dict)
        dat_dict[year] = push!(dat_dict[year],lang) 
        # note that push! is not our favorite thing to do in Julia, 
        # but we're focusing on correctness rather than speed here
    else
        dat_dict[year] = [lang]
    end
end

# Alternatively:

curyear = dat_read_dataframe.year[1]

dat_dict[curyear] = [dat_read_dataframe.language[1]]

for (i,nextyear) in enumerate(dat_read_dataframe.year[2:end])
    if nextyear == curyear
        #same key
        dat_dict[curyear] = push!(dat_dict[curyear], dat_read_dataframe.language[i+1])
        # note that push! is not our favorite thing to do in Julia, 
        # but we're focusing on correctness rather than speed here
    else
        curyear = nextyear
        dat_dict[curyear] = [dat_read_dataframe.language[i+1]]
    end
end

@show length(keys(dat_dict))
@show length(unique(dat_read[:,1]))

# Revisiting functions.

function year_created(dat_dict, language::String)
    keys_vec = collect(keys(dat_dict))
    lookup = map(keyid -> findfirst(dat_dict[keyid].==language), keys_vec)
    # now the lookup vector has `nothing` or a numeric value. We want to find the index of the numeric value.
    return keys_vec[findfirst((!isnothing).(lookup))]
end

year_created(dat_dict,"Julia")

how_many_per_year(dat_dict, year::Int64) = length(dat_dict[year])
how_many_per_year(dat_dict, 2011)

# Note function dropmissing for dealing with missing values, and keyword missing for missing. 

```

## Linear Algebra


```julia
# Get packages.

using LinearAlgebra
using SparseArrays
using Images
using MAT

# Get a random matrix, then mess around.

A = rand(10,10)
Atranspose = A'
A = A*Atranspose
@show A[11] == A[1,2]

b = rand(10)
x = A \ b
@show norm(A * x - b)

# Note:
#  A is of type Matrix, 
#  b is of type Vector, 
#  Atranspose is of type Adjoint (?), 
#  operator \ is better for solving than function inv.

@show typeof(A)
@show typeof(b)
@show typeof(rand(1, 10))
@show typeof(Atranspose)

# Note equivalences of objects across types, but mind dimensions.

Matrix{Float64} == Array{Float64,2}
Vector{Float64} == Array{Float64,1}

# Note access to parent matrix through adjoint.

Atranspose.parent

B = copy(Atranspose) # Recall Julia's handling of identities and copies.

# Check out longer treatment of operator \ for linear magic. 

# Shifting into factorizations of linear systems. 

# Consider L * U = P * A

luA = lu(A)

norm(luA.L * luA.U - luA.P*A)


# Consider Q * R = A.

qrA = qr(A)

norm(qrA.Q * qrA.R - A)

# Consider Cholesky factorization for a symmetric positive definite matrix.

isposdef(A)

cholA = cholesky(A)

norm(cholA.L * cholA.U - A)

cholA.L
cholA.U

factorize(A)

Diagonal([1,2,3]) # As an example of a diagonal. 

I(3) # Note that I() is a function.

# Shifting into linear algebra for sparse matrices.

S = sprand(5,5,2/5)

S.rowval

Matrix(S)

S.colptr
S.m

# Skipping images as matrices.

# Note that this all deals with made-up matrices and doesn't really get into empirical matrix problems (except sparseness).

```

## Statistics


```julia
# Handle packages.

using Statistics
using StatsBase
using RDatasets
using Plots
using StatsPlots
using KernelDensity
using Distributions
using LinearAlgebra
using HypothesisTests
using PyCall
using MLBase

D = dataset("datasets", "faithful")
@show names(D)
D

describe(D)

eruptions = D[!,:Eruptions]
scatter(eruptions, label = "Eruptions")
waittime = D[!,:Waiting]
scatter!(waittime, label = "Wait Time")

# This isn't useful yet.

# Try some basic statistical plots.

boxplot(["Eruption Length"], eruptions, legend = false, size=(200,400), whisker_width = 1, ylabel = "Time in Minutes")

histogram(eruptions, label = "Eruptions")
histogram(eruptions, bins = :sqrt, label = "Eruptions")

# Try some density plots.

p = kde(eruptions)

histogram(eruptions, label = "Eruptions")
plot!(p.x, p.density .* length(eruptions), linewidth = 3, color = 2, label="KDE Fit") # Note transformation for interpretable density.

histogram(eruptions, bins = :sqrt, label = "Eruptions")
plot!(p.x, p.density .* length(eruptions) .*0.2, linewidth = 3,color = 2,label = "KDE Fit") # Ditto.

# Try comparisons with random distributions.

myrandomvector = randn(100_000)
histogram(myrandomvector)
p = kde(myrandomvector)
plot!(p.x, p.density .* length(myrandomvector) .*0.1, linewidth = 3, color = 2, label = "KDE Fit")

d = Normal()
myrandomvector = rand(d, 100000)
histogram(myrandomvector)
p = kde(myrandomvector)
plot!(p.x, p.density .* length(myrandomvector) .*0.1, linewidth = 3,color = 2, label = "KDE Fit") 

b = Binomial(40) 
myrandomvector = rand(b, 1000000)
histogram(myrandomvector)
p = kde(myrandomvector)
plot!(p.x, p.density .* length(myrandomvector) .*0.5, color = 2,label = "KDE Fit")

# Try fitting with function fit().

x = rand(1000)
d = fit(Normal, x)
myrandomvector = rand(d, 1000)
histogram(myrandomvector, nbins=20, fillalpha = 0.3,label = "Fit")
histogram!(x, nbins = 20, linecolor = :red, fillalpha = 0.3, label = "Vector")

x = eruptions
d = fit(Normal, x)
myrandomvector = rand(d, 1000)
histogram(myrandomvector, nbins = 20, fillalpha = 0.3)
histogram!(x, nbins=20, linecolor = :red, fillalpha = 0.3)

# Into hypothesis testing. Mind p-value methods.

myrandomvector = randn(1000)
OneSampleTTest(myrandomvector)
OneSampleTTest(eruptions)

# Stuck on package management issues for Python.

# scipy_stats = pyimport("scipy.stats")
# @show scipy_stats.spearmanr(eruptions,waittime)
# @show scipy_stats.pearsonr(eruptions,waittime)
# scipy_stats.pearsonr(eruptions,waittime)

corspearman(eruptions, waittime)

cor(eruptions,waittime)

scatter(eruptions, waittime, xlabel = "Eruption Length",
    ylabel = "Wait Time between Eruptions", legend = false, grid = false, size = (400, 300))

# AUC and Confusion Matrix (?) with MLBase

gt = [1, 1, 1, 1, 1, 1, 1, 2]
pred = [1, 1, 2, 2, 1, 1, 1, 1]
C = confusmat(2, gt, pred)   # compute confusion matrix
C ./ sum(C, dims=2)   # normalize per class
sum(diag(C)) / length(gt)  # compute correct rate from confusion matrix
correctrate(gt, pred)
C = confusmat(2, gt, pred)   

gt = [1, 1, 1, 1, 1, 1, 1, 0];
pred = [1, 1, 0, 0, 1, 1, 1, 1]
ROC = MLBase.roc(gt,pred)
recall(ROC)
precision(ROC)
```

## Dimensionality Reduction


```julia
# Skipping this. Note PCA, t-SNE, and UMAP.
```

## Clustering


```julia
# Handle packages.

using Clustering
using VegaLite
using VegaDatasets
using DataFrames
using Statistics
using JSON
using CSV
using Distances

# Get real estate data.

download("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv","newhouses.csv")
houses = CSV.read("newhouses.csv", DataFrame)

# Check it out, then visualize.

names(houses)

cali_shape = JSON.parsefile("data/california-counties.json")
VV = VegaDatasets.VegaJSONDataset(cali_shape,"data/california-counties.json")

@vlplot(width=500, height=300) +
@vlplot(
    mark={
        :geoshape,
        fill=:black,
        stroke=:white
    },
    data={
        values=VV,
        format={
            type=:topojson,
            feature=:cb_2015_california_county_20m
        }
    },
    projection={type=:albersUsa},
)+
@vlplot(
    :circle,
    data=houses,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size={value=12},
    color="median_house_value:q"
                    
)

# Try some bins.

bucketprice = Int.(div.(houses[!,:median_house_value],50000))
insertcols!(houses,3,:cprice=>bucketprice)

@vlplot(width=500, height=300) +
@vlplot(
    mark={
        :geoshape,
        fill=:black,
        stroke=:white
    },
    data={
        values=VV,
        format={
            type=:topojson,
            feature=:cb_2015_california_county_20m
        }
    },
    projection={type=:albersUsa},
)+
@vlplot(
    :circle,
    data=houses,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size={value=12},
    color="cprice:n"
)

# Try k-means.

X = houses[!, [:latitude,:longitude]]
C = kmeans(Matrix(X)', 10) 
insertcols!(houses,3,:cluster10=>C.assignments)

@vlplot(width=500, height=300) +
@vlplot(
    mark={
        :geoshape,
        fill=:black,
        stroke=:white
    },
    data={
        values=VV,
        format={
            type=:topojson,
            feature=:cb_2015_california_county_20m
        }
    },
    projection={type=:albersUsa},
)+
@vlplot(
    :circle,
    data=houses,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size={value=12},
    color="cluster10:n"    
)

# Try k-medoids with pairwise Euclidean distances.

xmatrix = Matrix(X)'
D = pairwise(Euclidean(), xmatrix, xmatrix,dims=2) 

K = kmedoids(D,10)
insertcols!(houses,3,:medoids_clusters=>K.assignments)

@vlplot(width=500, height=300) +
@vlplot(
    mark={
        :geoshape,
        fill=:black,
        stroke=:white
    },
    data={
        values=VV,
        format={
            type=:topojson,
            feature=:cb_2015_california_county_20m
        }
    },
    projection={type=:albersUsa},
)+
@vlplot(
    :circle,
    data=houses,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size={value=12},
    color="medoids_clusters:n"          
)

# Try hierarchical clustering.

K = hclust(D)
L = cutree(K;k=10)
insertcols!(houses,3,:hclust_clusters=>L)

@vlplot(width=500, height=300) +
@vlplot(
    mark={
        :geoshape,
        fill=:black,
        stroke=:white
    },
    data={
        values=VV,
        format={
            type=:topojson,
            feature=:cb_2015_california_county_20m
        }
    },
    projection={type=:albersUsa},
)+
@vlplot(
    :circle,
    data=houses,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size={value=12},
    color="hclust_clusters:n"             
)

# Try DBSCAN clustering.

using Distances
dclara = pairwise(SqEuclidean(), Matrix(X)',dims=2)
L = dbscan(dclara, 0.05, 10)
@show length(unique(L.assignments))

insertcols!(houses,3,:dbscanclusters3=>L.assignments)

@vlplot(width=500, height=300) +
@vlplot(
    mark={
        :geoshape,
    
        fill=:black,
        stroke=:white
    },
    data={
        values=VV,
        format={
            type=:topojson,
            feature=:cb_2015_california_county_20m
        }
    },
    projection={type=:albersUsa},
)+
@vlplot(
    :circle,
    data=houses,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size={value=12},
    color="dbscanclusters3:n"     
)
```

## Classification 


```julia
# Get packages.

using GLMNet
using RDatasets
using MLBase
using Plots
using DecisionTree
using Distances
using NearestNeighbors
using Random
using LinearAlgebra
using DataStructures
using LIBSVM

# Pick an accuracy function.

findaccuracy(predictedvals,groundtruthvals) = sum(predictedvals.==groundtruthvals)/length(groundtruthvals)

# Get data.

iris = dataset("datasets", "iris")

# Manipulate data.

X = Matrix(iris[:,1:4])
irislabels = iris[:,5]

irislabelsmap = labelmap(irislabels)
y = labelencode(irislabelsmap, irislabels)

# Split data.

function perclass_splits(y,at)
    uids = unique(y)
    keepids = []
    for ui in uids
        curids = findall(y.==ui)
        rowids = randsubseq(curids, at) 
        push!(keepids,rowids...)
    end
    return keepids
end

trainids = perclass_splits(y,0.7)
testids = setdiff(1:length(y),trainids)

# Pick a prediction function.

assign_class(predictedvalue) = argmin(abs.(predictedvalue .- [1,2,3]))

# Try out some methods:

#  1. Lasso

path = glmnet(X[trainids, :], y[trainids])
cv = glmnetcv(X[trainids, :], y[trainids])
mylambda = path.lambda[argmin(cv.meanloss)]

path = glmnet(X[trainids, :], y[trainids], lambda = [mylambda]);

q = X[testids,:];
predictions_lasso = GLMNet.predict(path,q)

predictions_lasso = assign_class.(predictions_lasso)
findaccuracy(predictions_lasso,y[testids])

#  2. Ridge

# Note that the only difference is the change in alpha (to zero).

path = glmnet(X[trainids,:], y[trainids],alpha=0);
cv = glmnetcv(X[trainids,:], y[trainids],alpha=0)
mylambda = path.lambda[argmin(cv.meanloss)]
path = glmnet(X[trainids,:], y[trainids],alpha=0,lambda=[mylambda]);
q = X[testids,:];
predictions_ridge = GLMNet.predict(path,q)
predictions_ridge = assign_class.(predictions_ridge)
findaccuracy(predictions_ridge,y[testids])

#  3. Elastic Net

# As above, only alpha is 0.50.

path = glmnet(X[trainids,:], y[trainids],alpha=0.5);
cv = glmnetcv(X[trainids,:], y[trainids],alpha=0.5)
mylambda = path.lambda[argmin(cv.meanloss)]
path = glmnet(X[trainids,:], y[trainids],alpha=0.5,lambda=[mylambda]);
q = X[testids,:];
predictions_EN = GLMNet.predict(path,q)
predictions_EN = assign_class.(predictions_EN)
findaccuracy(predictions_EN,y[testids])

#  4. Decision Tree

# With package DecisionTree

model = DecisionTreeClassifier(max_depth=2)
DecisionTree.fit!(model, X[trainids,:], y[trainids])
q = X[testids,:];
predictions_DT = DecisionTree.predict(model, q)
findaccuracy(predictions_DT,y[testids])

#  5. Random Forest

# Also in DecisionTree

model = RandomForestClassifier(n_trees=20)
DecisionTree.fit!(model, X[trainids,:], y[trainids])
q = X[testids,:];
predictions_RF = DecisionTree.predict(model, q)
findaccuracy(predictions_RF,y[testids])

#  6. Nearest Neighbor

# With NearestNeighbor

Xtrain = X[trainids,:]
ytrain = y[trainids]
kdtree = KDTree(Xtrain')
queries = X[testids,:]
idxs, dists = knn(kdtree, queries', 5, true)
c = ytrain[hcat(idxs...)]
possible_labels = map(i->counter(c[:,i]),1:size(c,2))
predictions_NN = map(i->parse(Int,string(string(argmax(possible_labels[i])))),1:size(c,2))
findaccuracy(predictions_NN,y[testids])

#  7. Support Vector Machines

# With LIBSVM

Xtrain = X[trainids,:]
ytrain = y[trainids]
model = svmtrain(Xtrain', ytrain)
predictions_SVM, decision_values = svmpredict(model, X[testids,:]')
findaccuracy(predictions_SVM,y[testids])

# Compare results.

overall_accuracies = zeros(7)
methods = ["lasso","ridge","EN", "DT", "RF","kNN", "SVM"]
ytest = y[testids]
overall_accuracies[1] = findaccuracy(predictions_lasso,ytest)
overall_accuracies[2] = findaccuracy(predictions_ridge,ytest)
overall_accuracies[3] = findaccuracy(predictions_EN,ytest)
overall_accuracies[4] = findaccuracy(predictions_DT,ytest)
overall_accuracies[5] = findaccuracy(predictions_RF,ytest)
overall_accuracies[6] = findaccuracy(predictions_NN,ytest)
overall_accuracies[7] = findaccuracy(predictions_SVM,ytest)
hcat(methods, overall_accuracies)

# Note that all methods should work well, and all methods except decision trees and random forests should be accurate (perfect).
```

## Regression


```julia
using Plots
using Statistics
using StatsBase
using PyCall
using DataFrames
using GLM
using Tables
using XLSX
using MLBase
using RDatasets
using LsqFit

xvals = repeat(1:0.5:10, inner = 2)
yvals = 3 .+ xvals .+ 2 .* rand(length(xvals)) .-1
scatter(xvals, yvals, color=:black, leg=false)

# This is not an obvious approach to the intuition for a regression, but who am I to judge?

function find_best_fit(xvals,yvals)
    meanx = mean(xvals)
    meany = mean(yvals)
    stdx = std(xvals)
    stdy = std(yvals)
    r = cor(xvals,yvals)
    a = r*stdy/stdx
    b = meany - a*meanx
    return a,b
end

a,b = find_best_fit(xvals,yvals)
ynew = a .* xvals .+ b

np = pyimport("numpy"); # hey wait just a damn minute

xdata = xvals
ydata = yvals
@time myfit = np.polyfit(xdata, ydata, 1);
ynew2 = collect(xdata) .* myfit[1] .+ myfit[2];
scatter(xvals,yvals)
plot!(xvals,ynew)
plot!(xvals,ynew2)

data = DataFrame(X=xdata, Y=ydata)
ols = lm(@formula(Y ~ X), data)
plot!(xdata,predict(ols))

# Whatever. Shifting into linear regression on real data.

R = XLSX.readxlsx("data/zillow_data_download_april2020.xlsx")

sale_counts = R["Sale_counts_city"][:]
df_sale_counts = DataFrame(sale_counts[2:end,:],Symbol.(sale_counts[1,:]))

monthly_listings = R["MonthlyListings_City"][:]
df_monthly_listings = DataFrame(monthly_listings[2:end,:],Symbol.(monthly_listings[1,:]))

monthly_listings_2020_02 = df_monthly_listings[!,[1,2,3,4,5,end]]
rename!(monthly_listings_2020_02, Symbol("2020-02") .=> Symbol("listings"))

sale_counts_2020_02 = df_sale_counts[!,[1,end]]
rename!(sale_counts_2020_02, Symbol("2020-02") .=> Symbol("sales"))

# Unclear what the goal of this block is besides an array of plots, but note the array of plots!

Feb2020data = innerjoin(monthly_listings_2020_02,sale_counts_2020_02,on=:RegionID) #, type="outer")
dropmissing!(Feb2020data)
sales = Feb2020data[!,:sales]
# prices = Feb2020data[!,:price]
counts = Feb2020data[!,:listings]
using DataStructures
states = Feb2020data[!,:StateName]
C = counter(states)
C.map
countvals = values(C.map)
topstates = sortperm(collect(countvals),rev=true)[1:10]
states_of_interest = collect(keys(C.map))[topstates]
all_plots = Array{Plots.Plot}(undef,10)
for (i,si) in enumerate(states_of_interest)
    curids = findall(Feb2020data[!,:StateName].==si)
    data = DataFrame(X=float.(counts[curids]), Y=float.(sales[curids]))
    ols = GLM.lm(@formula(Y ~ 0 + X), data)    
    all_plots[i] = scatter(counts[curids],sales[curids],markersize=2,
        xlim=(0,500),ylim=(0,500),color=i,aspect_ratio=:equal,
        legend=false,title=si)
    @show si,coef(ols)
    plot!(counts[curids],predict(ols),color=:black)
end
plot(all_plots...,layout=(2,5),size=(900,300))

# And use of plot() with a for() loop.

plot()
for (i,si) in enumerate(states_of_interest)
    curids = findall(Feb2020data[!,:StateName].==si)
    data = DataFrame(X=float.(counts[curids]), Y=float.(sales[curids]))
    ols = GLM.lm(@formula(Y ~ 0 + X), data)    
    scatter!(counts[curids],sales[curids],markersize=2,
        xlim=(0,500),ylim=(0,500),color=i,aspect_ratio=:equal,
        legend=false,marker=(3,3,stroke(0)),alpha=0.2)
        if si == "NC" || si == "CA" || si == "FL"
            annotate!([(500-20,10+coef(ols)[1]*500,text(si,10))])
        end
    @show si,coef(ols)
    plot!(counts[curids],predict(ols),color=i,linewidth=2)
end
# plot(all_plots...,layout=(2,5),size=(900,300))
xlabel!("listings")
ylabel!("sales")

# Into logistic regression.

data = DataFrame(X=[1,2,3,4,5,6,7], Y=[1,0,1,1,1,1,1])
linear_reg = lm(@formula(Y ~ X), data)
scatter(data[!,:X],data[!,:Y],legend=false,size=(300,200))
plot!(1:7,predict(linear_reg))

cats = dataset("MASS", "cats")

lmap = labelmap(cats[!,:Sex])
ci = labelencode(lmap, cats[!,:Sex])
scatter(cats[!,:BWt],cats[!,:HWt],color=ci,legend=false)

lmap

data = DataFrame(X=cats[!,:HWt], Y=ci.-1)
probit = glm(@formula(Y ~ X), data, Binomial(), LogitLink())
scatter(data[!,:X],data[!,:Y],label="ground truth gender",color=6)
scatter!(data[!,:X],predict(probit),label="predicted gender",color=7)

xvals = 0:0.05:10
yvals = 1*exp.(-xvals*2) + 2*sin.(0.8*pi*xvals) + 0.15 * randn(length(xvals));
scatter(xvals,yvals,legend=false)

@. model(x, p) = p[1]*exp(-x*p[2]) + p[3]*sin(0.8*pi*x)
p0 = [0.5, 0.5, 0.5]
myfit = curve_fit(model, xvals, yvals, p0)

p = myfit.param
findyvals = p[1]*exp.(-xvals*p[2]) + p[3]*sin.(0.8*pi*xvals)
scatter(xvals,yvals,legend=false)
plot!(xvals,findyvals)

# Point being, a regression can handle any nonsense the user throws into it.

# And linear regression is sometimes not a great fit (visually).

@. model(x, p) = p[1]*x
myfit = curve_fit(model, xvals, yvals, [0.5])
p = myfit.param
findyvals = p[1]*xvals
scatter(xvals,yvals,legend=false)
plot!(xvals,findyvals)
```

## Graphs


```julia
using LightGraphs
using MatrixNetworks
using VegaDatasets
using DataFrames
using SparseArrays
using LinearAlgebra
using Plots
using VegaLite

airports = dataset("airports")
flightsairport = dataset("flights-airport")

airports

flightsairportdf = DataFrame(flightsairport)

allairports = vcat(flightsairportdf[!,:origin],flightsairportdf[!,:destination])
uairports = unique(allairports)

# create an airports data frame that has a subset of airports that are only included in the routes dataset
airportsdf = DataFrame(airports)
subsetairports = map(i->findfirst(airportsdf[!, :iata].==uairports[i]),1:length(uairports))
airportsdf_subset = airportsdf[subsetairports,:]

# build the adjacency matrix
ei_ids = findfirst.(isequal.(flightsairportdf[!,:origin]), [uairports])
ej_ids = findfirst.(isequal.(flightsairportdf[!,:destination]), [uairports])
edgeweights = flightsairportdf[!,:count]
;
A = sparse(ei_ids,ej_ids,1,length(uairports),length(uairports))
A = max.(A,A')

# What!

spy(A)

issymmetric(A)

# Note that LightGraphs does the heavy lifting for graphs in Julia, but other packages can be handy (or simpler).

L = SimpleGraph(A)

G=SimpleGraph(10) #SimpleGraph(nnodes,nedges) 
add_edge!(G,7,5)#modifies graph in place.
add_edge!(G,3,5)
add_edge!(G,5,2)

cc = scomponents(A)

degrees = sum(A,dims=2)[:]
p1 = plot(sort(degrees,rev=true),ylabel="log degree",legend=false,yaxis=:log)
p2 = plot(sort(degrees,rev=true),ylabel="degree",legend=false)
plot(p1,p2,size=(600,300))

maxdegreeid = argmax(degrees)
uairports[maxdegreeid]

us10m = dataset("us-10m")
@vlplot(width=500, height=300) +
@vlplot(
    mark={
        :geoshape,
        fill=:lightgray,
        stroke=:white
    },
    data={
        values=us10m,
        format={
            type=:topojson,
            feature=:states
        }
    },
    projection={type=:albersUsa},
) +
@vlplot(
    :circle,
    data=airportsdf_subset,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size={value=10},
    color={value=:steelblue}
)+
@vlplot(
    :rule,
    data=flightsairport,
    transform=[
        {filter={field=:origin,equal=:ATL}},
        {
            lookup=:origin,
            from={
                data=airportsdf_subset,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["origin_latitude", "origin_longitude"]
        },
        {
            lookup=:destination,
            from={
                data=airportsdf_subset,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["dest_latitude", "dest_longitude"]
        }
    ],
    projection={type=:albersUsa},
    longitude="origin_longitude:q",
    latitude="origin_latitude:q",
    longitude2="dest_longitude:q",
    latitude2="dest_latitude:q"
)

# Analytical demo with shortest paths by Dijkstra's algorithm.

ATL_paths = dijkstra(A,maxdegreeid)

ATL_paths[1][maxdegreeid]

maximum(ATL_paths[1])

@show stop1 = argmax(ATL_paths[1])
@show uairports[stop1]
;

@show stop2 = ATL_paths[2][stop1]
@show uairports[stop2]
;

@show stop3 = ATL_paths[2][stop2]
@show uairports[stop3]
;

@show stop4 = ATL_paths[2][stop3]
@show uairports[stop4]
;

using VegaLite, VegaDatasets

us10m = dataset("us-10m")
airports = dataset("airports")

@vlplot(width=800, height=500) +
@vlplot(
    mark={
        :geoshape,
        fill="#eee",
        stroke=:white
    },
    data={
        values=us10m,
        format={
            type=:topojson,
            feature=:states
        }
    },
    projection={type=:albersUsa},
) +
@vlplot(
    :circle,
    data=airportsdf_subset,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size={value=5},
    color={value=:gray}
) +
@vlplot(
    :line,
    data={
        values=[
            {airport=:ATL,order=1},
            {airport=:SEA,order=2},
            {airport=:JNU,order=3},
            {airport=:GST,order=4}
        ]
    },
    transform=[{
        lookup=:airport,
        from={
            data=airports,
            key=:iata,
            fields=["latitude","longitude"]
        }
    }],
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    order={field=:order,type=:ordinal}
)

nodeid = argmin(degrees)
@show uairports[nodeid]
d = dijkstra(A,nodeid)
argmax(d[1]),uairports[argmax(d[1])]

function find_path(d,id)
    shortestpath = zeros(Int,1+Int.(d[1][id]))
    shortestpath[1] = id
    for i = 2:length(shortestpath)
        shortestpath[i] = d[2][shortestpath[i-1]]
    end
    return shortestpath
end
p = find_path(d,123)
uairports[p]

# Set-up for a Minimum Spanning Tree by Prim's algorithm

?mst_prim

ti,tj,tv,nverts = mst_prim(A)

df_edges = DataFrame(:ei=>uairports[ti],:ej=>uairports[tj])

@vlplot(width=800, height=500) +
@vlplot(
    mark={
        :geoshape,
        fill="#eee",
        stroke=:white
    },
    data={
        values=us10m,
        format={
            type=:topojson,
            feature=:states
        }
    },
    projection={type=:albersUsa},
) +
@vlplot(
    :circle,
    data=airportsdf_subset,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size={value=20},
    color={value=:gray}
) +
@vlplot(
    :rule,
    data=df_edges, #data=flightsairport,
    transform=[
        {
            lookup=:ei,
            from={
                data=airportsdf_subset,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["originx", "originy"]
        },
        {
            lookup=:ej,
            from={
                data=airportsdf_subset,
                key=:iata,
                fields=["latitude", "longitude"]
            },
            as=["destx", "desty"]
        }
    ],
    projection={type=:albersUsa},
    longitude="originy:q",
    latitude="originx:q",
    longitude2="desty:q",
    latitude2="destx:q"
)

# Playing with PageRank

?MatrixNetworks.pagerank

v = MatrixNetworks.pagerank(A,0.85)

sum(v)

insertcols!(airportsdf_subset,7,:pagerank_value=>v)

@vlplot(width=500, height=300) +
@vlplot(
    mark={
        :geoshape,
        fill="#eee",
        stroke=:white
    },
    data={
        values=us10m,
        format={
            type=:topojson,
            feature=:states
        }
    },
    projection={type=:albersUsa},
) +
@vlplot(
    :circle,
    data=airportsdf_subset,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size="pagerank_value:q",
    color={value=:steelblue}
)

# Clustering Coefficients

?clustercoeffs

cc = clustercoeffs(A)
cc[findall(cc.<=eps())] .= 0
cc

insertcols!(airportsdf_subset,7,:ccvalues=>cc)

@vlplot(width=500, height=300) +
@vlplot(
    mark={
        :geoshape,
        fill="#eee",
        stroke=:white
    },
    data={
        values=us10m,
        format={
            type=:topojson,
            feature=:states
        }
    },
    projection={type=:albersUsa},
) +
@vlplot(
    :circle,
    data=airportsdf_subset,
    projection={type=:albersUsa},
    longitude="longitude:q",
    latitude="latitude:q",
    size="ccvalues:q",
    color={value=:gray}
)
```

## Numerical Optimization


```julia
# Skipping. The examples shown are really interesting but not quite helpful.
```

## Neural Nets


```julia
# Skipping.
```

## Other Languages


```julia
# Python

using PyCall

math = pyimport("math")
math.sin(math.pi / 4)

python_networkx = pyimport("networkx")

py"""
import numpy
def find_best_fit_python(xvals,yvals):
    meanx = numpy.mean(xvals)
    meany = numpy.mean(yvals)
    stdx = numpy.std(xvals)
    stdy = numpy.std(yvals)
    r = numpy.corrcoef(xvals,yvals)[0][1]
    a = r*stdy/stdx
    b = meany - a*meanx
    return a,b
"""

xvals = repeat(1:0.5:10, inner=2)
yvals = 3 .+ xvals .+ 2 .* rand(length(xvals)) .-1
find_best_fit_python = py"find_best_fit_python"
a,b = find_best_fit_python(xvals,yvals)

# R

using RCall

r = rcall(:sum, Float64[1.0, 4.0, 6.0])

typeof(r[1])

# Yoink an object from the Julia environment into an R environment, sort of?

z = 1
@rput z # Does this have a Python equivalent?

r = R"z+z"

r[1]

x = randn(10)

# Yoink a function from an R environment into the Julia environment, sort of?

@rimport base as rbase
rbase.sum([1, 2, 3])

# Run $install.packages("boot") in the REPL for the following to work.

@rlibrary boot

R"t.test($x)"

# Showing the Julia equivalent of the preceding lines:

using HypothesisTests
OneSampleTTest(x)

# C

t = ccall(:clock, Int32, ())
```

## Visualization


```julia
# Using Plots.jl with backend gr().

ENV["GKS_ENCODING"] = "utf-8"

# This is ridiculous.

stateabbreviations = Dict("Alabama" => "AL",
    "Alaska" => "AK",
    "Arizona" => "AZ",
    "Arkansas" => "AR",
    "California" => "CA",
    "Colorado" => "CO",
    "Connecticut" => "CT",
    "Delaware" => "DE",
    "Florida" => "FL",
    "Georgia" => "GA",
    "Hawaii" => "HI",
    "Idaho" => "ID",
    "Illinois" => "IL",
    "Indiana" => "IN",
    "Iowa" => "IA",
    "Kansas" => "KS",
    "Kentucky" => "KY",
    "Louisiana" => "LA",
    "Maine" => "ME",
    "Maryland" => "MD",
    "Massachusetts" => "MA",
    "Michigan" => "MI",
    "Minnesota" => "MN",
    "Mississippi" => "MS",
    "Missouri" => "MO",
    "Montana" => "MT",
    "Nebraska" => "NE",
    "Nevada" => "NV",
    "New Hampshire" => "NH",
    "New Jersey" => "NJ",
    "New Mexico" => "NM",
    "New York" => "NY",
    "North Carolina" => "NC",
    "North Dakota" => "ND",
    "Ohio" => "OH",
    "Oklahoma" => "OK",
    "Oregon" => "OR",
    "Pennsylvania" => "PA",
    "Rhode Island" => "RI",
    "South Carolina" => "SC",
    "South Dakota" => "SD",
    "Tennessee" => "TN",
    "Texas" => "TX",
    "Utah" => "UT",
    "Vermont" => "VT",
    "Virginia" => "VA",
    "Washington" => "WA",
    "West Virginia" => "WV",
    "Wisconsin" => "WI",
    "Wyoming" => "WY", 
    "District of Columbia"=>"DC");

using Plots
using StatsPlots # this package provides stats specific plotting functions
gr()

using Statistics
using StatsBase
using MLBase

# Note function shenanigans to get nice labels. Unclear if this is still an issue worth working around.

xtickslabels = ["one","five","six","fourteen"]
p = plot(rand(15),xticks = ([1,5,6,14],xtickslabels),xrotation=90,xtickfont=font(13))

function pad_empty_plot(p)
    ep = plot(grid=false,legend=false,axis=false,framestyle = :box)#empty plot
    newplot = plot(p,ep,layout=@layout([a{0.99h};b{0.001h}]))
    return newplot
end
pad_empty_plot(p)

using XLSX
using DataFrames
D = DataFrame(XLSX.readtable("data/zillow_data_download_april2020.xlsx", "Sales_median_price_city")...);
dropmissing!(D)
states = D[:,:StateName];

states = D[:,:StateName];

NYids = findall(states.=="New York")
NYframe = dropmissing(D[NYids,:])
CAids = findall(states.=="California")
CAframe = dropmissing(D[CAids,:])
FLids = findall(states.=="Florida")
FLframe = dropmissing(D[FLids,:])

# Violin Plots

# pick a year: 2020-02
ca = CAframe[!,Symbol("2020-02")]
ny = NYframe[!,Symbol("2020-02")]
fl = FLframe[!,Symbol("2020-02")]

violin(["New York"], ny,legend=false,alpha=0.8)
violin!(["California"], ca,alpha=0.8)
violin!(["Florida"],fl,alpha=0.8)

# 2020 data
ca = CAframe[!,Symbol("2020-02")]
ny = NYframe[!,Symbol("2020-02")]
fl = FLframe[!,Symbol("2020-02")]
violin(["New York"], ny,legend=false,alpha=0.8,side=:right)
violin!(["California"], ca,alpha=0.8,side=:right)
violin!(["Florida"],fl,alpha=0.8,side=:right)

### get the February 2010 data
ca10 = CAframe[!,Symbol("2010-02")]
ny10 = NYframe[!,Symbol("2010-02")]
fl10 = FLframe[!,Symbol("2010-02")]

violin!(["New York"], ny10,legend=false,alpha=0.8,side=:left)
violin!(["California"], ca10,alpha=0.8,side=:left)
violin!(["Florida"],fl10,alpha=0.8,side=:left)

# pick a year: 2019-02
ca = CAframe[!,Symbol("2010-02")]
ny = NYframe[!,Symbol("2010-02")]
fl = FLframe[!,Symbol("2010-02")]
violin(["New York"], ny,alpha=0.8,side=:left,color=6,label="2010-02")
violin!(["California"], ca,alpha=0.8,side=:left,color=6,label="")
violin!(["Florida"],fl,alpha=0.8,side=:left,color=6,label="")

# pick a year: 2020-02
ca = CAframe[!,Symbol("2020-02")]
ny = NYframe[!,Symbol("2020-02")]
fl = FLframe[!,Symbol("2020-02")]
violin!(["New York"], ny,alpha=0.8,side=:right,color=7,label="2020-02")
violin!(["California"], ca,alpha=0.8,side=:right,color=7,label="")
violin!(["Florida"],fl,alpha=0.8,side=:right,color=7,label="")

# pick a year: 2019-02
ca = CAframe[!,Symbol("2010-02")]
ny = NYframe[!,Symbol("2010-02")]
fl = FLframe[!,Symbol("2010-02")]
violin(["New York"], ny,alpha=0.8,side=:left,color=6,label="2010-02")
violin!(["California"], ca,alpha=0.8,side=:left,color=6,label="")
violin!(["Florida"],fl,alpha=0.8,side=:left,color=6,label="")

# pick a year: 2020-02
ca = CAframe[!,Symbol("2020-02")]
ny = NYframe[!,Symbol("2020-02")]
fl = FLframe[!,Symbol("2020-02")]
violin!(["New York"], ny,alpha=0.8,side=:right,color=7,label="2020-02")
violin!(["California"], ca,alpha=0.8,side=:right,color=7,label="")
violin!(["Florida"],fl,alpha=0.8,side=:right,color=7,label="")

m = median(ny)
ep = 0.1
annotate!([(0.5+ep,m+0.05,text(m/1000,10,:left))])

m = median(ca)
ep = 0.1
annotate!([(1.5+ep,m+0.05,text(m/1000,10,:left))])

m = median(fl)
ep = 0.1
annotate!([(2.5+ep,m+0.05,text(m/1000,10,:left))])

plot!(xtickfont=font(10),size=(500,300))

# putting it together.

ep = 0.05 # will later be used in padding for annotations

# set up the plot
plot(xtickfont=font(10))

states_of_interest = ["New York", "California", "Florida", "Ohio","Idaho"]
years_of_interst = [Symbol("2010-02"),Symbol("2020-02")]

# year 1
xstart = 0.5
yi = years_of_interst[1]
for si in states_of_interest
    curids = findall(states.==si)
    curFrame = D[curids,:]
    curprices = curFrame[!,yi]
    m = median(curprices)
    annotate!([(xstart-ep,m+0.05,text(m/1000,8,:right))])
    xstart += 1
    violin!([si], curprices,alpha=0.8,side=:left,color=6,label="")
end
plot!(Shape([],[]),color=6,label=yi)

# year 2
xstart = 0.5
yi = years_of_interst[2]
for si in states_of_interest
    curids = findall(states.==si)
    curFrame = D[curids,:]
    curprices = curFrame[!,yi]
    m = median(curprices)
    annotate!([(xstart+ep,m+0.05,text(m/1000,8,:left))])
    xstart += 1
    violin!([si], curprices,alpha=0.8,side=:right,color=7,label="")
end
plot!(Shape([],[]),color=7,label=yi)
ylabel!("housing prices")

# Bars and Histograms

mapstates = labelmap(states)
stateids = labelencode(mapstates, states)
histogram(stateids,nbins=length(mapstates))

# first we'll start with sorting
h = fit(Histogram, stateids,nbins=length(mapstates))
sortedids = sortperm(h.weights,rev=true)
bar(h.weights[sortedids],legend=false)

bar(h.weights[sortedids],legend=false,orientation = :horizontal,yflip=true)

# just an example of annotations
bar(h.weights[sortedids],legend=false,orientation = :horizontal,yflip=true,size=(400,500))
stateannotations = mapstates.vs[sortedids]
for i = 1:3
    annotate!([(h.weights[sortedids][i]-5,i,text(stateannotations[i],10,:left))])
end
plot!()

bar(h.weights[sortedids],legend=false,orientation = :horizontal,yflip=true,linewidth=0,width=0,size=(400,500))
stateannotations = mapstates.vs[sortedids]
for i = 1:length(stateannotations)
    annotate!([(h.weights[sortedids][i]-5,i,text(stateabbreviations[stateannotations[i]],5,:left))])
end
plot!()

bar(h.weights[sortedids],legend=false,orientation = :horizontal,
        yflip=true,linewidth=0,width=0,color=:gray,alpha=0.8)
stateannotations = mapstates.vs[sortedids]
for i = 20:20:200
    plot!([i,i],[50,0],color=:white)
end
for i = 1:length(stateannotations)
    annotate!([(h.weights[sortedids][i]-5,i,text(stateabbreviations[stateannotations[i]],6,:left))])
end
plot!(grid=false,yaxis=false,xlim=(0,maximum(h.weights)),xticks = 0:20:200)
xlabel!("number of listings")

bar(h.weights[sortedids],legend=false,orientation = :horizontal,
        yflip=true,linewidth=0,color=:gray,alpha=0.8,size=(300,500))
stateannotations = mapstates.vs[sortedids]
ht = length(h.weights)
for i = 20:20:200
    plot!([i,i],[ht,0],color=:white)
end
for i = 1:length(stateannotations)
    annotate!([(h.weights[sortedids][i]+2,i,text(stateabbreviations[stateannotations[i]],6,:left))])
end
plot!(grid=false,yaxis=false,xlim=(0,maximum(h.weights)+5),xticks = 0:20:200)
xlabel!("number of listings")

f = Plots.plot!(inset = bbox(0.7,0.15,0.25,0.6,:top,:left))
bar!(f[2],h.weights[sortedids][21:end],legend=false,orientation = :horizontal,
        yflip=true,linewidth=0,width=0,color=:gray,alpha=0.8)
for i = 21:length(stateannotations)
    annotate!(f[2],[(h.weights[sortedids][i]+1,i-20,text(stateabbreviations[stateannotations[i]],6,:left))])
end
plot!(f[2],[10,10],[20,0],color=:white,xticks=0:10:20,yaxis=false,grid=false,xlim=(0,20))
plot!()

# Plots w/ Error

M = Matrix(NYframe[:,5:end])

xtickslabels = string.(names(NYframe[!,5:end]))

plot()
for i = 1:size(M,1)
    plot!(M[i,:],legend=false)
end
plot!()
p = plot!(xticks = (1:4:length(xtickslabels),xtickslabels[1:4:end]),xrotation=90,xtickfont=font(8),grid=false)
pad_empty_plot(p)

function find_percentile(M, pct)
    r = zeros(size(M,2))
    for i = 1:size(M,2)
        v = M[:,i]
        len = length(v)
        ind = floor(Int64,pct*len)
        newarr = sort(v);
        r[i] = newarr[ind];
    end
    return r
end

md = find_percentile(M,0.5)
mx = find_percentile(M,0.8)
mn = find_percentile(M,0.2)
plot(md,ribbon =(md.-mn,mx.-md),color = :blue,label="NY",grid=false)
p = plot!(xticks = (1:4:length(xtickslabels),xtickslabels[1:4:end]),xrotation=90,xtickfont=font(8))
pad_empty_plot(p)

function plot_individual_state!(plotid,statevalue,colorid)
    curids = findall(states.==statevalue)
    curFrame = D[curids,:]
    M = Matrix(curFrame[:,5:end])
    md = find_percentile(M,0.5)
    mx = find_percentile(M,0.8)
    mn = find_percentile(M,0.2)
    plot!(plotid,md,ribbon =(md.-mn,mx.-md),color = colorid,label=stateabbreviations[statevalue],grid=false)
    plot!(plotid,xticks = (1:4:length(xtickslabels),xtickslabels[1:4:end]),xrotation=90,xtickfont=font(8))
end

plotid = plot()
plot_individual_state!(plotid,"Indiana",1)
plot_individual_state!(plotid,"Ohio",2)
plot_individual_state!(plotid,"Idaho",3)
# plot_individual_state!(plotid,"California",4)
ylabel!("prices")
pad_empty_plot(plotid)

vector1 = rand(10)
vector2 = rand(10)*100
plot(vector1,label = "b",size=(300,200))
plot!(twinx(), vector2,color=2,axis=false)

xtickslabels = NYframe[!,:RegionName]

sz = NYframe[!,:SizeRank]
pc = NYframe[!,end]
M = Matrix(NYframe[:,5:end])
M = copy(M')
md = find_percentile(M,0.9)

md = find_percentile(M,0.5)
mx = find_percentile(M,0.9)
mn = find_percentile(M,0.1)
vector1 = sz

plot()
plot!(md,ribbon =(md.-mn,mx.-md),color = 1,grid=false,label="")

plot!(xticks = (1:length(xtickslabels),xtickslabels),xrotation=90,xtickfont=font(10))
plot!(twinx(), vector1,color=2,label="",ylabel="rank",grid=false,xticks=[],linewidth=2)
plot!(Shape([0], [0]),color=1,label="Prices (left)")
p = plot!([],[],color=2,label="Rank (right)")
ep = plot(grid=false,legend=false,axis=false,framestyle = :box)#empty plot
plot(p,ep,layout=@layout([a{0.85h};b{0.001h}]))

# High-Dimensional Plots

CA202002 = CAframe[!,Symbol("2020-02")]
CA201002 = CAframe[!,Symbol("2010-02")]
scatter(CA201002,CA202002)

CA202002 = CAframe[!,Symbol("2020-02")]
CA201002 = CAframe[!,Symbol("2010-02")]
CAranks = CAframe[!,:SizeRank]
scatter(CA201002,CA202002,legend=false,markerstrokewidth=0,markersize=3,alpha=0.6,grid=false)

using ColorSchemes

# normalize the ranks to be between 0 and 1
continuousranks = CAranks./maximum(CAranks)

# create a placeholder vector that will store the color of each value
colorsvec = Vector{RGB{Float64}}(undef,length(continuousranks))

# and finally map the colors according to ColorSchemes.autumn1, there are many other schemes you can choose from
map(i->colorsvec[i]=get(ColorSchemes.autumn1,continuousranks[i]),1:length(colorsvec))

continuousdates = CAranks./maximum(CAranks)
colorsvec = Vector{RGB{Float64}}(undef,length(continuousdates))
map(i->colorsvec[i]=get(ColorSchemes.autumn1,continuousdates[i]),1:length(colorsvec))
scatter(CA201002,CA202002,color=colorsvec,
    legend=false,markerstrokewidth=0,markersize=3,grid=false)
xlabel!("2010-02 prices",xguidefontsize=10)
ylabel!("2020-02 prices",yguidefontsize=10)
p1 = plot!()

#set up the plot canvas
xvals = 0:100
s = Shape([0,1,1,0],[0,0,1,1])
plot(s,color=ColorSchemes.autumn1[1],grid=false,axis=false,
    legend=false,linewidth=0,linecolor=nothing)

for i = 2:101
    s = Shape([xvals[i],xvals[i]+1,xvals[i]+1,xvals[i]],[0,0,1,1])
    plot!(s,color=ColorSchemes.autumn1[i],grid=false,axis=false,
    legend=false,linewidth=0,linecolor=nothing)
end

mynormalizer = maximum(CAranks)
xtickslabels = 0:div(mynormalizer,10):mynormalizer
continuousdates = xtickslabels./mynormalizer
xticksloc = round.(Int,continuousdates.*101)

# annotate using the ranks
rotatedfont = font(10, "Helvetica",rotation=90)
for i = 1:length(xtickslabels)
    annotate!(xticksloc[i],0.5,text(xtickslabels[i], rotatedfont))
end
p2 = plot!()

mylayout = @layout([a{0.89h};b{0.1h}])
plot(p1,p2,layout=mylayout)

# Wrapping up: neat, essential visualizations; note Julia's approach to layering snippets to produce a chart. 
```
