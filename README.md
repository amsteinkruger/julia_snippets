This document compiles code used in free courses at https://juliaacademy.com/.

# 0, 1


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

# 2


```julia
myfavoriteanimals = ("penguins", "cats", "sugargliders") # is a tuple
myfavoriteanimals[1]
# since tuples are immutable, we can't update it
```
