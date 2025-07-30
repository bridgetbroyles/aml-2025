#----------------------------#

student_scores = {
    "Alice": 85,
    "Bob": 92,
    "Charlie": 78,
    "Dana": 82,
    "Emily": 95
}

names = ["Alice", "Brian", "Charlie", "David", "Emily", "Fred"]

for name in names:
    if name in student_scores:
        print(name, student_scores[name])
    else:
        student_scores[name] = 0
        print(f"{name} added")

print(student_scores)

#----------------------------#

student_scores = {
    "Alice": [85, 87, 92, 96],
    "Bob": [92, 91, 94, 84],
    "Charlie": [78, 80, 82, 84]
}

# Print Bob's grade on assignment 2 (index 1)
print("Bob's grade on assignment 2:", student_scores["Bob"][1])

# Change Charlie's grade on assignment 3 (index 2) to an 8
student_scores["Charlie"][2] = 8

# Optional: print to confirm the change
print("Updated Charlie's grades:", student_scores["Charlie"])
