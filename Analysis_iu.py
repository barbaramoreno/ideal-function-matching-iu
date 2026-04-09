""" Analysis: in this part we will select the ideal functions with the best match according to the instructions,
the steps are the following
-read and load the data
-find the best ideal functions using Least-Square
-Clasify the test data using sqrt
-save the results in sql

References from Programming with Python – CSEMDSPWP01 Course Book. IU:
- Unit 2: "Classes and Inheritance"
- Unit 3: "Scientific Calculations"
- Unit 4: "Handling and Raising Exceptions"
- Unit 4: "User-Defined Exceptions"

"""

import pandas as pd
import numpy as np

# Custom Exceptions
# Reference: Unit 4 "User-Defined Exceptions"
# these are our own error classes, they inherit from Exception
# so we can raise them when something goes wrong with our data

class DataLoadingError(Exception):
    """
    Custom exception for when data cant be loaded
    for example when a CSV file is missing or has a wrong format
    """
    def __init__(self, message="Error loading the data file"):
        self.message = message
        super().__init__(self.message)


class MappingError(Exception):
    """
    Custom exception for when a test point cant be mapped
    to any of the four chosen ideal functions
    """
    def __init__(self, x_value, y_value):
        self.x_value = x_value
        self.y_value = y_value
        self.message = (
            f"The point (x={x_value}, y={y_value}) could not be "
            f"mapped to any ideal function"
        )
        super().__init__(self.message)

# Base Class
# Ref: Unit 2

class DataAnalyze:
    """
    This is the main class for analyze the data
    it is linked with FunctionMatch y TestClassifier(inherit)
    also handles validation of the df input
    """

    def __init__(self, data_main):
        """
        Constructor of the DataAnalyze class (base)
        Args:
            data_main: main df to analyze the data
        Raises:
            DataLoadingError: if the input is not a valid DataFrame
        """
        # validate that the input is actually a dataframe
        if not isinstance(data_main, pd.DataFrame):
            raise DataLoadingError("Input must be a pandas DataFrame")
        if data_main.empty:
            raise DataLoadingError("DataFrame is empty")
        self.data = data_main

    def get_values(self, column_name):
        """
        Obtain values from dataframe with array
        Args:
            column_name: column name
        Returns:
            numpy array with the values
        Raises:
            KeyError: if column doesnt exist
        """
        if column_name not in self.data.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame")
        return self.data[column_name].values


# Function Matching (Least-Square)
# Ref: Unit 2 and 3

class FunctionMatch(DataAnalyze):
    """
    Finds the best 4 functions, using Least-Square
    inherits from DataAnalyze class
    compares each training function against all 50 ideal functions
    and picks the one with the smallest error
    """

    def __init__(self, train_df, ideal_df):
        """
        Constructor
        Args:
            train_df: training data ['x', 'y1', 'y2', 'y3', 'y4']
            ideal_df: 50 ideal functions ['x', 'y1' to 'y50']
        Raises:
            DataLoadingError: if df are invalid or x values dont match
        """
        super().__init__(train_df)  # inherits call
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.matched_functions = {}      # {train_col: ideal_col}
        self.max_dev_values = {}    # {ideal_col: max_deviation}

        # check that both df have the same x values
        # if not, comparing them makes no sense
        if not np.array_equal(
            self.train_df['x'].values, self.ideal_df['x'].values
        ):
            raise DataLoadingError(
                "Training and ideal data must have the same x-values"
            )

    def calc_leastsq(self, train_col, ideal_col):
        """
        Calculate least-squares ( sum((y_train - y_ideal)^2)
        - values are converted to numpy array
        - (a - b)^2 for each element
        - np.sum() sums all
        Uses NumPy vectorized operations (Unit 3)
        Args:
            train_col: name of the training column ['x', 'y1', 'y2', 'y3', 'y4']
            ideal_col: name of the ideal column ['x', 'y1' to 'y50']
        Returns:
            sum of deviations
        """
        train_values = self.train_df[train_col].values
        ideal_values = self.ideal_df[ideal_col].values
        return np.sum((train_values - ideal_values) ** 2)

    def best_match(self, train_col):
        """
        find the best match between ideal and training data
        loops through all 50 ideal functions and picks the one with lowest error
        Args:
            train_col: name of the training column ['x', 'y1', 'y2', 'y3', 'y4']
        Returns:
            tuple: (name of best ideal, least error)
        """
        # ideal columns = all except 'x'

        ideal_columns = [c for c in self.ideal_df.columns if c.lower() != 'x']

        lsq_errors = {}
        for ideal_col in ideal_columns:
                lsq_errors[ideal_col] = self.calc_leastsq(train_col, ideal_col)
        best = min(lsq_errors, key=lsq_errors.get)
        return best, lsq_errors[best]

    def match_all(self):
        """
        Find the best 4 match of ideal functions
        Also calculates the max deviation then for the sqrt criteria
        Returns:
            dict: {column_training: column_ideal}
        """

        train_cols = [c for c in self.train_df.columns if c.lower() != 'x']

        for train_col in train_cols:
            best_ideal, error = self.best_match(train_col)
            self.matched_functions[train_col] = best_ideal

            # calculate max deviation (needed for test classification step)
            train_vals = self.train_df[train_col].values
            ideal_vals = self.ideal_df[best_ideal].values
            self.max_dev_values[best_ideal] = np.max(
                np.abs(train_vals - ideal_vals)
            )

            print(f"  {train_col} → {best_ideal} "
                  f"(LSQ error: {error:.2f}, "
                  f"max dev: {self.max_dev_values[best_ideal]:.4f})")

        return self.matched_functions

# Test Data Classification (sqrt(2) criterion)
# Ref: Unit 2,3 and 4

class TestClassifier(DataAnalyze):
    """
    Classifies test data points against the 4 chosen ideal functions
    using the sqrt(2) deviation criterion
    inherits from DataAnalyze class

    a test point (x, y) gets assigned to an ideal function if:
    y_test is within max_deviation * sqrt(2) of y_ideal(x).
    if it passes the criteria, we save it with the deviation
    Otherwise raise a MappingError
    """

    def __init__(self, test_df, ideal_df, matched_functions, max_dev_values):
        """
        Constructor
        Args:
            test_df: test data with columns [x, y]
            ideal_df: all 50 ideal functions
            matched_functions: dict from FunctionMatch {train_col: ideal_col}
            max_dev_values: dict from FunctionMatch {ideal_col: max_dev}
        """
        super().__init__(test_df)  # inherits call
        self.test_df = test_df
        self.ideal_df = ideal_df
        self.matched_functions = matched_functions
        self.max_dev_values = max_dev_values
        self.classified_points = []

    def classify_point(self, x_val, y_val):
        """
        Classify a single test point (x, y) against the 4 ideal functions
        checks if the deviation stays within the allowed threshold (max_dev * sqrt(2))
        if multiple functions match, selects the one with smallest deviation
        Args:
            x_val: x-value of the test point
            y_val: y-value of the test point
        Returns:
            dict with {x, y, delta_y, ideal_func} if mapped, None if not
        """
        # find the row in ideal_df where x matches our test point
        ideal_row = self.ideal_df[self.ideal_df['x'] == x_val]

        if ideal_row.empty:
            return None

        closest_delta = None
        closest_func = None

        # check each of the 4 chosen ideal functions
        for train_col, ideal_col in self.matched_functions.items():
            y_ideal = ideal_row[ideal_col].values[0]
            delta = abs(y_val - y_ideal)
            # the threshold is max_deviation * sqrt(2) according to instructions
            allowed_dev = self.max_dev_values[ideal_col] * np.sqrt(2)

            if delta <= allowed_dev:
                # if its within threshold and better than previous match it is saved
                if closest_delta is None or delta < closest_delta:
                    closest_delta = delta
                    closest_func = ideal_col

        if closest_func is not None:
            return {
                'x': x_val,
                'y': y_val,
                'delta_y': closest_delta,
                'ideal_func': closest_func
            }
        return None

    def classify_all(self):
        """
        Classify all test data points
        iterates row-by-row
        Returns:
            DataFrame with columns [x, y, delta_y, ideal_func]
        """
        assigned = 0
        not_assigned = 0

        for _, row in self.test_df.iterrows():
            x_val = row['x']
            y_val = row['y']

            # exception handling
            try:
                result = self.classify_point(x_val, y_val)

                if result is not None:
                    self.classified_points.append(result)
                    assigned += 1
                else:
                    not_assigned += 1
                    try:
                        raise MappingError(x_val, y_val)
                    except MappingError as e:
                        print(f"  Note: {e}")

            except Exception as e:
                print(f"  Unexpected error at x={x_val}: {e}")

        print(f"\n  Results: {assigned} mapped, "
              f"{not_assigned} not mapped")

        if self.classified_points:
            return pd.DataFrame(self.classified_points)
        else:
            return pd.DataFrame(
                columns=['x', 'y', 'delta_y', 'ideal_func']
            )
