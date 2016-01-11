"""
Printing
Use str() or append string using ,
"""
for x in range(10):
    print(x," ", end="")
print()

"""
Lists
"""
print("Lists\n")
num_list = []

for i in range(10):
    num_list.append(i+1)

for x in num_list:
    print(str(x) + " ", end="")
print()

# Concatenate lists
appended_list = [0] + num_list

for x in appended_list:
    print(str(x) + " ", end="")
print()

# Length of lists
print("Length of Appended List : " + str(len(appended_list)))

"""
Dictionary
"""
print("\nDictionary\n")
# Create a dict
table = {}
#dictionary = dict()

# Set values using keys
table[0] = "Hello"
table[1] = "World"

print(table)

# Check if key exists
does_key_3_exist = 3 in table
print(does_key_3_exist)

# Or avoid errors with the get() method
val_3 = table.get(3, -1)
print(val_3)

"""
Default Dict
"""
print("\nDefault Dict\n")
from collections import defaultdict
defaultDict = defaultdict(int)

# If key does not exist, uses default value of provided object. Here default of int it 0

# Especially helpful with lists
listDict = defaultdict(list)

"""
Counters
"""
print("\nCounters\n")
from collections import Counter
counter = Counter([0, 1, 2, 0, 3, 3, 3])
print(counter)

# Another way to print
for num, cont in counter.most_common(3):
    print(str(num) + "->" + str(cont), end=",")
print()

"""
Sets
"""
print("\nSets\n")
setvals = set()
setvals.add(1)
setvals.add(2)
setvals.add(2)
print("No of elements in set : " + str(len(setvals)))

# List to Set
list2 = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 7]
for x in list2:
    print(str(x) + " ", end="")
print("List count : " + str(len(list2)))

num_set = set(list2)
print(num_set)

"""
Comprehension of Lists
"""
print("\nList Comprehensions\n")

lc = [(x, y)
      for x in range(10)
      for y in range(10)]

print("Length of lc : ", len(lc))
for x, y in lc:
    print("(", x, ",", y, ")", end=" ")
print()


"""
Generators / Iterators
"""
print("\nGenerators/Iterators\n")

# Generator of even numbers
def evens(uptil):
    i = 2
    while i < uptil:
        yield i
        i += 2

for x in evens(10):
    print(x, " ", end=" ")
print()