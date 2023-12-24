 

+
-
Play


05:15
09:01
Mute

Settings
cognitiveclass.ai logo
Introduction to Pandas for Data Analysis
Estimated time: 10 Mins
Objective:
Learn what Pandas Series are and how to create them.
Understand how to access and manipulate data within a Series.
Discover the basics of creating and working with Pandas DataFrames.
Learn how to access, modify, and analyze data in DataFrames.
Gain insights into common DataFrame attributes and methods.
What is Pandas?
Pandas is a popular open-source data manipulation and analysis library for the Python programming language. It provides a powerful and flexible set of tools for working with structured data, making it a fundamental tool for data scientists, analysts, and engineers.
Pandas is designed to handle data in various formats, such as tabular data, time series data, and more, making it an essential part of the data processing workflow in many industries.

Here are some key features and functionalities of Pandas:

Data Structures: Pandas offers two primary data structures - DataFrame and Series.

A DataFrame is a two-dimensional, size-mutable, and potentially heterogeneous tabular data structure with labeled axes (rows and columns).
A Series is a one-dimensional labeled array, essentially a single column or row of data.
Data Import and Export: Pandas makes it easy to read data from various sources, including CSV files, Excel spreadsheets, SQL databases, and more. It can also export data to these formats, enabling seamless data exchange.

Data Merging and Joining: You can combine multiple DataFrames using methods like merge and join, similar to SQL operations, to create more complex datasets from different sources.

Efficient Indexing: Pandas provides efficient indexing and selection methods, allowing you to access specific rows and columns of data quickly.

Custom Data Structures: You can create custom data structures and manipulate data in ways that suit your specific needs, extending Pandas' capabilities.

Importing Pandas:
Import Pandas using the import command, followed by the library's name.
Commonly, Pandas is imported as pd for brevity in code.

1
import pandas as pd
Copied!
Data Loading:
Pandas can be used to load data from various sources, such as CSV and Excel files.
The read_csv function is used to load data from a CSV file into a Pandas DataFrame.
To read a CSV (Comma-Separated Values) file in Python using the Pandas library, you can use the pd.read_csv() function. Here's the syntax to read a CSV file:

1
2
3
4
import pandas as pd
# Read the CSV file into a DataFrame
df = pd.read_csv('your_file.csv')
Copied!
Replace 'your_file.csv' with the actual file path of your CSV file. Make sure that the file is located in the same directory as your Python script, or you provide the correct file path.

What is a Series?
A Series is a one-dimensional labeled array in Pandas. It can be thought of as a single column of data with labels or indices for each element. You can create a Series from various data sources, such as lists, NumPy arrays, or dictionaries
Here's a basic example of creating a Series in Pandas:

1
2
3
4
5
6
7
import pandas as pd
# Create a Series from a list
data = [10, 20, 30, 40, 50]
s = pd.Series(data)
print(s)
Copied!
In this example, we've created a Series named s with numeric data. Notice that Pandas automatically assigned numerical indices (0, 1, 2, 3, 4) to each element, but you can also specify custom labels if needed.

Accessing Elements in a Series
You can access elements in a Series using the index labels or integer positions. Here are a few common methods for accessing Series data:

Accessing by label
1
print(s[2])     # Access the element with label 2 (value 30)
Copied!
Accessing by position
1
print(s.iloc[3]) # Access the element at position 3 (value 40)
Copied!
Accessing multiple elements
1
print(s[1:4])   # Access a range of elements by label
Copied!
Series Attributes and Methods
Pandas Series come with various attributes and methods to help you manipulate and analyze data effectively. Here are a few essential ones:

values: Returns the Series data as a NumPy array.
index: Returns the index (labels) of the Series.
shape: Returns a tuple representing the dimensions of the Series.
size: Returns the number of elements in the Series.
mean(), sum(), min(), max(): Calculate summary statistics of the data.
unique(), nunique(): Get unique values or the number of unique values.
sort_values(), sort_index(): Sort the Series by values or index labels.
isnull(), notnull(): Check for missing (NaN) or non-missing values.
apply(): Apply a custom function to each element of the Series.
What is a DataFrames?
A DataFrame is a two-dimensional labeled data structure with columns of potentially different data types. Think of it as a table where each column represents a variable, and each row represents an observation or data point. DataFrames are suitable for a wide range of data, including structured data from CSV files, Excel spreadsheets, SQL databases, and more.

Creating DataFrames from Dictionaries:
DataFrames can be created from dictionaries, with keys as column labels and values as lists representing rows.

1
2
3
4
5
6
7
8
9
10
11
import pandas as pd
# Creating a DataFrame from a dictionary
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 28],
        'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']}
df = pd.DataFrame(data)
print(df)
Copied!
Column Selection:
You can select a single column from a DataFrame by specifying the column name within double brackets.
Multiple columns can be selected in a similar manner, creating a new DataFrame.

1
print(df['Name'])  # Access the 'Name' column
Copied!
Accessing Rows:
You can access rows by their index using .iloc[] or by label using .loc[].

1
2
print(df.iloc[2])   # Access the third row by position
print(df.loc[1])    # Access the second row by label
Copied!
Slicing:
You can slice DataFrames to select specific rows and columns.

1
2
print(df[['Name', 'Age']])  # Select specific columns
print(df[1:3])             # Select specific rows
Copied!
Finding Unique Elements:
Use the unique method to determine the unique elements in a column of a DataFrame.

1
unique_dates = df['Age'].unique()
Copied!
Conditional Filtering:
You can filter data in a DataFrame based on conditions using inequality operators.
For instance, you can filter albums released after a certain year.

1
high_above_102 = df[df['Age'] > 25]
Copied!
Saving DataFrames:
To save a DataFrame to a CSV file, use the to_csv method and specify the filename with a “.csv” extension.Pandas provides other functions for saving DataFrames in different formats.

1
df.to_csv('trading_data.csv', index=False)
Copied!
DataFrame Attributes and Methods
DataFrames provide numerous attributes and methods for data manipulation and analysis, including:

shape: Returns the dimensions (number of rows and columns) of the DataFrame.
info(): Provides a summary of the DataFrame, including data types and non-null counts.
describe(): Generates summary statistics for numerical columns.
head(), tail(): Displays the first or last n rows of the DataFrame.
mean(), sum(), min(), max(): Calculate summary statistics for columns.
sort_values(): Sort the DataFrame by one or more columns.
groupby(): Group data based on specific columns for aggregation.
fillna(), drop(), rename(): Handle missing values, drop columns, or rename columns.
apply(): Apply a function to each element, row, or column of the DataFrame.
Pandas offers a wide range of methods beyond these examples. For more detailed information, please refer to the official documentation available on the Pandas official website.

Conclusion
In conclusion, mastering the use of Pandas Series and DataFrames is essential for effective data manipulation and analysis in Python. Series provide a foundation for handling one-dimensional data with labels, while DataFrames offer a versatile, table-like structure for working with two-dimensional data. Whether you're cleaning, exploring, transforming, or analyzing data, these Pandas data structures, along with their attributes and methods, empower you to efficiently and flexibly manipulate data to derive valuable insights. By incorporating Series and DataFrames into your data science toolkit, you'll be well-prepared to tackle a wide range of data-related tasks and enhance your data analysis capabilities.
To further your skills in data analysis with Pandas, consider the following next steps:

Practice:
Work with real datasets to apply what you've learned and gain hands-on experience.

Explore Documentation:
Visit the Pandas official website to explore the extensive documentation and discover more functions and methods.

Author
Akansha Yadav

Changelog
Date	Version	Changed by	Change Description
2023-10-02	1.0	Akansha Yadav	Created Reading
















Reading and writing files
Package/Method	Description	Syntax and Code Example
File opening modes	Different modes to open files for specific operations.	Syntax: r (reading) w (writing) a (appending) + (updating: read/write) b (binary, otherwise text)
1
Examples: with open("data.txt", "r") as file: content = file.read() print(content) with open("output.txt", "w") as file: file.write("Hello, world!") with open("log.txt", "a") as file: file.write("Log entry: Something happened.") with open("data.txt", "r+") as file: content = file.read() file.write("Updated content: " + content)</td>
Copied!
File reading methods	Different methods to read file content in various ways.	Syntax:
1
2
3
file.readlines() # reads all lines as a list
readline() # reads the next line as a string
file.read() # reads the entire file content as a string
Copied!
Example:

1
2
3
4
with open("data.txt", "r") as file:
    lines = file.readlines()
    next_line = file.readline()
    content = file.read()
Copied!
File writing methods	Different write methods to write content to a file.	Syntax:
1
2
file.write(content) # writes a string to the file
file.writelines(lines) # writes a list of strings to the file
Copied!
Example:

1
2
3
lines = ["Hello\n", "World\n"]
with open("output.txt", "w") as file:
    file.writelines(lines)
Copied!
Iterating over lines	Iterates through each line in the file using a `loop`.	Syntax:
1
for line in file: # Code to process each line
Copied!
Example:

1
2
with open("data.txt", "r") as file:
for line in file: print(line)
Copied!
Open() and close()	Opens a file, performs operations, and explicitly closes the file using the close() method.	Syntax:
1
2
file = open(filename, mode) # Code that uses the file
file.close()
Copied!
Example:

1
2
3
file = open("data.txt", "r")
content = file.read()
file.close()
Copied!
with open()	Opens a file using a with block, ensuring automatic file closure after usage.	Syntax:
1
with open(filename, mode) as file: # Code that uses the file
Copied!
Example:

1
2
with open("data.txt", "r") as file:
content = file.read()
Copied!
Pandas
Package/Method	Description	Syntax and Code Example
.read_csv()	Reads data from a `.CSV` file and creates a DataFrame.	Syntax: dataframe_name = pd.read_csv("filename.csv") Example: df = pd.read_csv("data.csv")
.read_excel()	Reads data from an Excel file and creates a DataFrame.	Syntax:
1
dataframe_name = pd.read_excel("filename.xlsx")
Copied!
Example:

1
df = pd.read_excel("data.xlsx")
Copied!
.to_csv()	Writes DataFrame to a CSV file.	Syntax:
1
dataframe_name.to_csv("output.csv", index=False)
Copied!
Example:

1
df.to_csv("output.csv", index=False)
Copied!
Access Columns	Accesses a specific column using [] in the DataFrame.	Syntax:
1
2
dataframe_name["column_name"] # Accesses single column
dataframe_name[["column1", "column2"]] # Accesses multiple columns
Copied!
Example:

1
2
df["age"]
df[["name", "age"]]
Copied!
describe()	Generates statistics summary of numeric columns in the DataFrame.	Syntax:
1
dataframe_name.describe()
Copied!
Example:

1
df.describe()
Copied!
drop()	Removes specified rows or columns from the DataFrame. axis=1 indicates columns. axis=0 indicates rows.	Syntax:
1
2
dataframe_name.drop(["column1", "column2"], axis=1, inplace=True)
dataframe_name.drop(index=[row1, row2], axis=0, inplace=True)
Copied!
Example:

1
2
df.drop(["age", "salary"], axis=1, inplace=True) # Will drop columns
df.drop(index=[5, 10], axis=0, inplace=True) # Will drop rows
Copied!
dropna()	Removes rows with missing NaN values from the DataFrame. axis=0 indicates rows.	Syntax:
1
dataframe_name.dropna(axis=0, inplace=True)
Copied!
Example:

1
df.dropna(axis=0, inplace=True)
Copied!
duplicated()	Duplicate or repetitive values or records within a data set.	Syntax:
1
dataframe_name.duplicated()
Copied!
Example:

1
duplicate_rows = df[df.duplicated()]
Copied!
Filter Rows	Creates a new DataFrame with rows that meet specified conditions.	Syntax:
1
filtered_df = dataframe_name[(Conditional_statements)]
Copied!
Example:

1
filtered_df = df[(df["age"] > 30) & (df["salary"] < 50000)
Copied!
groupby()	Splits a DataFrame into groups based on specified criteria, enabling subsequent aggregation, transformation, or analysis within each group.	Syntax:
1
2
grouped = dataframe_name.groupby(by, axis=0, level=None, as_index=True,
sort=True, group_keys=True, squeeze=False, observed=False, dropna=True)
Copied!
Example:

1
grouped = df.groupby(["category", "region"]).agg({"sales": "sum"})
Copied!
head()	Displays the first n rows of the DataFrame.	Syntax:
1
dataframe_name.head(n)
Copied!
Example:

1
df.head(5)
Copied!
Import pandas	Imports the Pandas library with the alias pd.	Syntax:
1
import pandas as pd
Copied!
Example:

1
import pandas as pd
Copied!
info()	Provides information about the DataFrame, including data types and memory usage.	Syntax:
1
dataframe_name.info()
Copied!
Example:

1
df.info()
Copied!
merge()	Merges two DataFrames based on multiple common columns.	Syntax:
1
merged_df = pd.merge(df1, df2, on=["column1", "column2"])
Copied!
Example:

1
merged_df = pd.merge(sales, products, on=["product_id", "category_id"])
Copied!
print DataFrame	Displays the content of the DataFrame.	Syntax:
1
print(df) # or just type df
Copied!
Example:

1
2
print(df)
df
Copied!
replace()	Replaces specific values in a column with new values.	Syntax:
1
dataframe_name["column_name"].replace(old_value, new_value, inplace=True)
Copied!
Example:

1
df["status"].replace("In Progress", "Active", inplace=True)
Copied!
tail()	Displays the last n rows of the DataFrame.	Syntax:
1
dataframe_name.tail(n)
Copied!
Example:

1
df.tail(5)
Copied!
Numpy
Package/Method	Description	Syntax and Code Example
Importing NumPy	Imports the NumPy library.	Syntax:
1
import numpy as np
Copied!
Example:

1
import numpy as np
Copied!
np.array()	Creates a one or multi-dimensional array,	Syntax:
1
2
array_1d = np.array([list1 values]) # 1D Array
array_2d = np.array([[list1 values], [list2 values]]) # 2D Array
Copied!
Example:

1
2
array_1d = np.array([1, 2, 3]) # 1D Array
array_2d = np.array([[1, 2], [3, 4]]) # 2D Array
Copied!
Numpy Array Attributes	- Calculates the mean of array elements
- Calculates the sum of array elements
- Finds the minimum value in the array
- Finds the maximum value in the array
- Computes dot product of two arrays	Example:
1
2
3
4
5
np.mean(array)
np.sum(array)
np.min(array
np.max(array)
np.dot(array_1, array_2)
Copied!
