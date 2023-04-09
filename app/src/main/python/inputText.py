# Import libraries
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import subprocess
import parse


# REST API

# Execute the Java file and retrieve the parameter value
cmd = ['java', '-cp', '.', 'ObjectID']
objectID = subprocess.check_output(cmd).decode().strip()

# Initialize Parse with your application ID and API key
parse.APPLICATION_ID = 'your_application_id'
parse.REST_API_KEY = '60hU03CUPV5GDjz8ycN4ukdeIB1skg67UipTLRd1'

# Define the class you want to query
MyClass = parse.Object.extend('Fully Green')

# Define your query criteria
query = parse.Query(MyClass)

# Execute the query and retrieve the results
input_test_X = query.find()

# SM Path of file
sm_language_file = "C:/Users/HP/Documents/CSUF Year 2021-2023/2 - Spring 2023/FH_2023/Data_Clean/sustainable_materials.csv"
sm_language_data = pd.read_csv(sm_language_file)

# Preprocess -- drop columns
sm_language_data = sm_language_data.drop(sm_language_data.columns[[0]], axis=1)
sm_language_data = sm_language_data.drop(sm_language_data.iloc[:, 2:5], axis=1)

# UM Path of file
um_language_file = "C:/Users/HP/Documents/CSUF Year 2021-2023/2 - Spring 2023/FH_2023/Data_Clean/unsustainable_materials.csv"
um_language_data = pd.read_csv(um_language_file)

# Preprocess -- drop columns
um_language_data = um_language_data.drop(um_language_data.columns[[0]], axis=1)
um_language_data = um_language_data.drop(um_language_data.iloc[:, 2:5], axis=1)

# Join datasets
sm_xm_language_data = pd.concat([sm_language_data, um_language_data], ignore_index=True)

# Create target objects
X = sm_xm_language_data['material_subcategories']
y = sm_xm_language_data['recyclable']

# Label encoding
le = LabelEncoder()
X_num = le.fit_transform(X)
X_num = pd.DataFrame(X_num, columns = ['material_subcategories'])

# Scalar transformation
ss = StandardScaler()
X_num = ss.fit_transform(X_num)
X_num = DataFrame(X_num, columns = ['material_subcategories'])

# Split into Train & Test (validate) data -- 70-30 distribution (categorical)
train_X_cat, test_X_cat, train_y_cat, test_y_cat = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Split into Train & Test (validate) data -- 70-30 distribution (numerical)
train_X_num, test_X_num, train_y_num, test_y_num = train_test_split(X_num, y, test_size = 0.2, random_state = 42)

# Calculate similarity between train and test -- Euclidean distance
# Sample test data point -- later user input
input_test_X = np.array(test_X_num)
input_test_X = input_test_X[0]
input_test_X_mat = np.array(test_X_cat)[0]
input_train_y_lab = np.array(train_y_cat)[0]

# Calculate Euclidean distance between testing data point and each training data point
distances = np.sqrt(np.sum((train_X_num - input_test_X)**2, axis=1))

# Find index of closest training data point
closest_index = np.argmin(distances)

# Print the testing and closest training data points together
print("Testing data point: ", input_test_X_mat)
print("Label of testing data point: ", input_train_y_lab)
print("Closest training data point: ", np.array(train_X_cat)[closest_index])
print("Label of closest training data point: ", np.array(train_y_cat)[closest_index])