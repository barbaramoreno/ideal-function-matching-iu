"""
Main module: this is the beginning, it connects everything together
the steps are:
1. the loading of the csv data (training, ideal, test)
2. storing the data in SQLite database using sqlalchemy
3. finding the 4 best ideal functions with Least-Square
4. classifying test data using sqrt(2) criterion
5. saving the results to database
6. generating the Bokeh visualizations
References from Programming with Python – CSEMDSPWP01 Course Book. IU:
-Unit 3 sections: "Scientific Calculations", "Data Visualization" and "Accessing Databases"
"""
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine
import os
import sys
# importing our classes and exceptions from Analysis_iu.py script
from Analysis_iu import (
    DataLoadingError,
    MappingError,
    FunctionMatch,
    TestClassifier
)
# importing the visualization function
from visualization_iu import create_all_plots

# Database Functions
# SQLAlchemy use it to create tables and store our dataframes

def create_database(db_name="results.db"):
    """
    Create a SQLite database connection using sqlalchemy
    Args:
        db_name: name of the database file
    Returns:
        sqlalchemy engine object
    Raises:
        DataLoadingError: if the database cant be created
    """
    try:
        engine = create_engine(f"sqlite:///{db_name}", echo=False)
        print(f"  Database created: {db_name}")
        return engine
    except Exception as e:
        raise DataLoadingError(f"Could not create database: {e}")

def load_csv(filepath):
    """
    Load a CSV file into a pd DataFrame
    Args:
        filepath: path to the CSV file
    Returns:
        pandas DataFrame with the loaded data
    Raises:
        DataLoadingError: if the file is missing or cant be parsed
    """
    try:
        df = pd.read_csv(filepath)
        print(f"  Loaded: {filepath} ({df.shape[0]} rows, {df.shape[1]} cols)")
        return df
    except FileNotFoundError:
        raise DataLoadingError(f"File not found: {filepath}")
    except pd.errors.ParserError:
        raise DataLoadingError(f"Could not read CSV file: {filepath}")

def store_db(engine, df, table_name):
    """
    Save a DataFrame to a table in the SQLite database
    uses pd to_sql which works with sqlalchemy engine
    Args:
        engine: sqlalchemy engine
        df: dataframe to save
        table_name: name of the table in the database
    Raises:
        DataLoadingError: if the data cant be saved
    """
    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"  Saved to DB table: '{table_name}' ({len(df)} rows)")
    except Exception as e:
        raise DataLoadingError(f"Could not save to database: {e}")

# Main Execution this function runs the whole workflow step by step

def main():
    """
    Main function that runs the complete workflow
    loads data, creates db, finds matches, classifies test, saves and plots
    """
    print("\n  Ideal Function Matching Task\n")

    # step 1: load the three csv files
    print("\n  Step 1: Loading CSV Data")
    try:
        train_df = load_csv("dataset/train.csv")
        ideal_df = load_csv("dataset/ideal.csv")
        test_df = load_csv("dataset/test.csv")
    except DataLoadingError as e:
        print(f"FATAL: {e}")
        sys.exit(1)

    # step 2, creating the sqlite database and store training and ideal data
    print("\n  Step 2: Creating the Database")
    try:
        engine = create_database("results.db")
        # table 1: training data (5 columns)
        store_db(engine, train_df, "training_data")
        # table 2: ideal functions (51 columns)
        store_db(engine, ideal_df, "ideal_functions")
    except DataLoadingError as e:
        print(f"FATAL: {e}")
        sys.exit(1)

    # step 3, finding the best 4 ideal functions using least-square
    print("\n  Step 3: Finding Best Ideal Functions (Least-Square)")
    try:
        func_matcher = FunctionMatch(train_df, ideal_df)
        matched_functions = func_matcher.match_all()
    except DataLoadingError as e:
        print(f"FATAL: {e}")
        sys.exit(1)

    print("\n  Summary of matches:")
    for train_col, ideal_col in matched_functions.items():
        dev = func_matcher.max_dev_values[ideal_col]
        print(f"    {train_col} --> {ideal_col} "
              f"(max deviation: {dev:.4f})")

    # step 4, classifying of each test point against the four chosen ideal functions
    print("\n  Step 4: Classifying Test Data")
    classifier = TestClassifier(
        test_df, ideal_df,
        matched_functions, func_matcher.max_dev_values
    )
    classification_df = classifier.classify_all()

    # step 5, saving the classification results to the database
    # table 3 from assignment, test results with required names
    print("\n  Step 5: Saving Results to Database")
    if not classification_df.empty:
        # renaming the columns to match the table structure from the task
        output_table = classification_df.rename(columns={
            'x': 'X (test func)',
            'y': 'Y (test func)',
            'delta_y': 'Delta Y (test func)',
            'ideal_func': 'No. of ideal func'
        })
        store_db(engine, output_table, "test_results")
    else:
        print("  No test points were mapped. Nothing to save.")

    # step 6, generating the bokeh plots

    print("\n  Step 6: Generating Visualizations")
    try:
        create_all_plots(
            train_df, ideal_df, test_df,
            matched_functions, classification_df
        )
    except Exception as e:
        print(f"  Visualization error: {e}")

    # final summary of everything
    print("\n  Task Completed Successfully")
    print(f"  Best matches: {matched_functions}")
    print(f"  Test points mapped: {len(classification_df)}/{len(test_df)}")
    print(f"  Database Results: results.db")
    print(f"  Plots: plots/ folder\n")

if __name__ == "__main__":
    main()
