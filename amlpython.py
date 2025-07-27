# Task 1.1
print("Hello Angie and Kai")

# Task 1.2
# print("Hello Angie and Kai")
print("Burnt orange and white,")
print("Longhorns stand with pride and might,")
print("Hook em, Texas fight!")

# Task 1.3
print()  # blank line
print("  @..@")
print(" (----)")
print("(>____<)")
print("^^~~~~^^")

# Task 2.1
student_name = "Me"
print(student_name, "was a dedicated computer science student at UT Austin.")
print("Each morning", student_name, "biked through the vibrant UT campus.")
print(student_name, "enjoyed all of the interesting CS classes.")
print("During the week,", student_name, "would meet up with friends to collaborate on challenging programming projects.")
print("On weekends,", student_name, "participated in hackathons.")
print("Computer science allowed", student_name, "to think logically and creatively.")

# Task 3.1
earth = 100
moon = earth * 0.1654
print(f"Earth weight: {earth}")
print(f"Moon weight: {moon:.2f}")

# Task 4.1
def say_hello(name): print("Hello", name, "!!!")
say_hello("Angie")
say_hello("Kai")
say_hello("Anshul")

# Task 4.2
def moon_weight(e): return e * 0.1654
print("Your moon weight is", round(moon_weight(150), 2), "lbs")

# Task 4.3
def triangle_area(b, h): return b * h / 2
print("Triangle area:", triangle_area(5, 8))

# Task 5.1b
s = "Hello World"
print(s[8])       # r
print(s[-2])      # l

# Task 5.2
def replace_item(lst, idx, val):
    lst[idx] = val
    return lst

# example
print(replace_item([1,2,3,4], 2, "hello"))  # [1, 2, "hello", 4]

# Task 5.3
lst = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
print(lst[2][1])   # 8

# Task 5.4
hotel = [
    ["available", "available", "available", "Angie",   "Brian"],   # 100–104
    ["Claire",    "available", "available", "available", "available"],  # 200–204
    ["available", "available", "David",    "available", "available"],  # 300–304
    ["available", "Emily",     "available", "available", "available"]   # 400–404
]

print(hotel[3][1])   # who’s in room 401?
print(hotel[1][4])   # who’s in room 204?

hotel[2][3] = "Frank"  # assign Frank to room 303

# Task 6.1
print(9 == "nine")    # False
var = 9
print(var == 9)       # True
print(3 <= 4)         # True example using <=
print(3 != 3)         # False example using !=

# Task 6.2
def paycheck(hours, rate):
    pay = hours * rate
    if hours > 40:
        pay += 100
    return pay

print(paycheck(38, 20))  # no overtime bonus
print(paycheck(45, 20))  # includes $100 bonus

# Task 6.3
def paycheck(hours, rate):
    pay = hours * rate
    if hours > 40:
        pay += 100
        if hours > 55:
            print("Too many hours worked!")
    return pay

print(paycheck(60, 20))  # prints warning and returns pay + bonus

# Task 7.1
for i in range(5):
    print(i)




