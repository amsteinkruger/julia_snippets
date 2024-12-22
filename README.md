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
# Something.
```
