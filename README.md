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
