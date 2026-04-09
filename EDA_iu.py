"""This is a script to perform exploratory data analysis very general"""

#importing the right library
import pandas as pd

#Reading the three files provided

train_df = pd.read_csv('dataset/train.csv')
ideal_df = pd.read_csv('dataset/ideal.csv')
test_df = pd.read_csv('dataset/test.csv')

#Train df exploration
print("train DF")
print(train_df.head())
print(f"columns: {list(train_df.columns)}")
train_df.info()
print(train_df.head())
#Name of each column
print(f"columns: {list(train_df.columns)}")
#size of the information rows and columns
print(f"size: {train_df.shape}")

#ideal df exploration
print("Ideal DF")
ideal_df.info()
print(ideal_df.head())
#Name of each column
print(f"columns: {list(ideal_df.columns)}")
#size of the information r x c
print(f"size: {ideal_df.shape}")

#test df exploration
print("Test DF")
test_df.info()
print(test_df.head())
#Name of each column
print(f"columns: {list(test_df.columns)}")
#size of the information r x c
print(f"size: {test_df.shape}")

