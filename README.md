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
