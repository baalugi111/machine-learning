nums = [3, 5, 7, 8, 12]
cubes = []

# Append the cubes of each number in nums to the cubes list
for num in nums:
  cube = num * num * num  # Calculate the cube of the number
  cubes.append(cube)
  
  dict = {}
dict['parrot'] = 2
dict['goat'] = 4
dict['spider'] = 8
dict['crab'] = 10

print(dict)

dict = {'parrot': 2, 'goat': 4, 'spider': 8, 'crab': 10}

total_legs = 0

for animal, legs in dict.items():
    print(f"{animal} has {legs} legs.")
    total_legs += legs

print(f"\nTotal legs of all animals: {total_legs}")



A = (3, 9, 4, [5, 6])

# Create a new list with the modified value
new_list = A[3].copy()  # Create a copy to avoid modifying the original list
new_list[0] = 8

# Create a new tuple with the modified list
B = (A[0], A[1], A[2], new_list)

print(B) 


B = ('a', 'p', 'p', 'l', 'e')
count = B.count('p')
print(count)


B = ('a', 'p', 'p', 'l', 'e')
index = B.index('l')
print(index)


import numpy as np

data = [[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]]

A = np.array(data)
print(A)

import numpy as np

# Assuming A is the NumPy array created earlier

C = np.zeros_like(A)
print(C)


import numpy as np

# Assuming A and z are defined as before

for i in range(A.shape[1]):
  C[:, i] = A[:, i] + z

print(C)


import numpy as np

X = np.array([[1, 2],
              [3, 4]])

Y = np.array([[5, 6],
              [7, 8]])

# Add the matrices X and Y
Z = X + Y

print(Z)


import numpy as np

Y = np.array([[5, 6],
              [7, 8]])

# Compute the element-wise square root of Y
Y_sqrt = np.sqrt(Y)

print(Y_sqrt)


import numpy as np

X = np.array([[1, 2],
              [3, 4]])

v = np.array([9, 10])

# Compute the dot product of X and v
result = X.dot(v)

print(result)


import numpy as np

X = np.array([[1, 2],
              [3, 4]])

# Compute the sum of each column
column_sums = np.sum(X, axis=0)

print(column_sums)


def Compute(distance, time):
    """Calculates velocity given distance and time.

    Args:
        distance: The distance traveled.
        time: The time taken to travel the distance.

    Returns:
        The velocity.
    """

    velocity = distance / time
    return velocity

# Example usage:
distance = 100  # meters
time = 10      # seconds
velocity = Compute(distance, time)
print("Velocity:", velocity, "m/s")


def mult(even_num):
  product = 1
  for num in even_num:
    product *= num
  return product

# Create a list of even numbers up to 12
even_num = [2, 4, 6, 8, 10, 12]

# Calculate the product of all even numbers
result = mult(even_num)

print("Product of even numbers:", result)



import pandas as pd

# Create a Pandas DataFrame
data = {'C1': [1, 2, 3, 5, 5],
        'C2': [6, 7, 5, 4, 8],
        'C3': [7, 9, 8, 6, 5],
        'C4': [7, 5, 2, 8, 8]}

pd = pd.DataFrame(data)

# Print the first two rows
print(pd.head(2))



import pandas as pd

# Create a Pandas DataFrame
data = {'C1': [1, 2, 3, 5, 5],
        'C2': [6, 7, 5, 4, 8],
        'C3': [7, 9, 8, 6, 5],
        'C4': [7, 5, 2, 8, 8]}

pd = pd.DataFrame(data)

# Print the second column
print(pd['C2'])


import pandas as pd

# Create a Pandas DataFrame
data = {'C1': [1, 2, 3, 5, 5],
        'C2': [6, 7, 5, 4, 8],
        'C3': [7, 9, 8, 6, 5],
        'C4': [7, 5, 2, 8, 8]}

pd = pd.DataFrame(data)

# Rename the third column
pd = pd.rename(columns={'C3': 'B3'})

print(pd)


import pandas as pd

# Create a Pandas DataFrame
data = {'C1': [1, 2, 3, 5, 5],
        'C2': [6, 7, 5, 4, 8],
        'C3': [7, 9, 8, 6, 5],
        'C4': [7, 5, 2, 8, 8]}

pd = pd.DataFrame(data)

# Rename the third column
pd = pd.rename(columns={'C3': 'B3'})

# Add a new column 'Sum'
pd['Sum'] = pd['C1'] + pd['C2'] + pd['B3'] + pd['C4']

print(pd)


import pandas as pd

# Create a Pandas DataFrame
data = {'C1': [1, 2, 3, 5, 5],
        'C2': [6, 7, 5, 4, 8],
        'C3': [7, 9, 8, 6, 5],
        'C4': [7, 5, 2, 8, 8]}

pd = pd.DataFrame(data)

# Rename the third column
pd = pd.rename(columns={'C3': 'B3'})

# Add a new column 'Sum' by summing each row
pd['Sum'] = pd.sum(axis=1)

print(pd)


import pandas as pd

# Assuming the CSV file is in the same directory or a specified path
file_path = "hello_sample.csv"  # Replace with the correct path

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

print(df)



print(df.tail(2))

print(df.info())

print(df.shape)


df.sort_values(by='Weight', inplace=True)
print(df)


 Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Print the DataFrame after dropping missing values
print(df)
Use code with caution.











