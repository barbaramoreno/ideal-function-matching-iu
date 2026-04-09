"""
Unit Tests for the ideal function matching task
tests all the main classes and functions to make sure they work correctly
uses the unittest framework
Reference from Programming with Python – CSEMDSPWP01 Course Book. IU:
-Unit 5 sections "Unit and Integration Testing" and "Documenting Code"
Runs with: python -m unittest test_analysis.py -v
"""
import unittest
import pandas as pd
import numpy as np
import os

from Analysis_iu import (
    DataLoadingError,
    MappingError,
    DataAnalyze,
    FunctionMatch,
    TestClassifier
)
from main_iu import load_csv, create_database, store_db

class TestCustomExceptions(unittest.TestCase):
    """
    Tests for our custom exception classes
    """

    def test_data_load_error_default_message(self):
        """DataLoadingError should have a default message when no arg is passed"""
        error = DataLoadingError()
        self.assertEqual(str(error), "Error loading the data file")

    def test_data_load_error_custom_message(self):
        """DataLoadingError should accept and show a custom message"""
        error = DataLoadingError("File not found")
        self.assertEqual(str(error), "File not found")

    def test_mapping_error_message(self):
        """MappingError should include the x and y values in the message"""
        error = MappingError(1.5, 3.2)
        self.assertIn("1.5", str(error))
        self.assertIn("3.2", str(error))

    def test_mapping_error_attributes(self):
        """MappingError should store x and y so we can access them later"""
        error = MappingError(2.0, 4.0)
        self.assertEqual(error.x_value, 2.0)
        self.assertEqual(error.y_value, 4.0)

class TestDataAnalyze(unittest.TestCase):
    """
    Tests for the DataAnalyze base class
    """

    def setUp(self):
        """creating a sample dataframe for testing"""
        self.sample_df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [2.0, 4.0, 6.0]
        })

    def test_valid_dataframe(self):
        """it should accept a valid DataFrame without errors"""
        data_handler = DataAnalyze(self.sample_df)
        self.assertIsNotNone(data_handler.data)

    def test_invalid_input_not_dataframe(self):
        """it should raise DataLoadingError if you pass a string instead of DataFrame"""
        with self.assertRaises(DataLoadingError):
            DataAnalyze("not a dataframe")

    def test_invalid_input_empty_dataframe(self):
        """it should raise DataLoadingError if the DataFrame is empty"""
        with self.assertRaises(DataLoadingError):
            DataAnalyze(pd.DataFrame())

    def test_get_values_valid_column(self):
        """get_values should return the correct numpy array"""
        data_handler = DataAnalyze(self.sample_df)
        values = data_handler.get_values('x')
        np.testing.assert_array_equal(values, np.array([1.0, 2.0, 3.0]))

    def test_get_values_invalid_column(self):
        """get_values should raise KeyError if the column dont exist"""
        data_handler = DataAnalyze(self.sample_df)
        with self.assertRaises(KeyError):
            data_handler.get_values('z')

class TestFunctionMatch(unittest.TestCase):
    """
    Tests for FunctionMatch class (Least-Square matching)
    we create fake data where we know the answer so we can verify it works
    """
    def setUp(self):
        """
        creating sample training and ideal data with known matches
        train y1=2x should match ideal y2=2x, etc
        """
        self.x_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        self.train_df = pd.DataFrame({
            'x': self.x_values,
            'y1': self.x_values * 2,          # y = 2x
            'y2': self.x_values ** 2,          # y = x^2
            'y3': np.sin(self.x_values),       # y = sin(x)
            'y4': self.x_values * 3 + 1        # y = 3x + 1
        })

        self.ideal_df = pd.DataFrame({
            'x': self.x_values,
            'y1': self.x_values * 5,           # y = 5x (no match for anything)
            'y2': self.x_values * 2,           # y = 2x (should match train y1)
            'y3': self.x_values ** 2,          # y = x^2 (should match train y2)
            'y4': np.sin(self.x_values),       # y = sin(x) (should match train y3)
            'y5': self.x_values * 3 + 1        # y = 3x+1 (should match train y4)
        })

    def test_calc_leastsq_perfect_match(self):
        """when two functions are identical the error should be 0"""
        func_matcher = FunctionMatch(self.train_df, self.ideal_df)
        error = func_matcher.calc_leastsq('y1', 'y2')  # both are 2x
        self.assertAlmostEqual(error, 0.0)

    def test_calc_leastsq_nonzero(self):
        """when functions are different the error should be greater than 0"""
        func_matcher = FunctionMatch(self.train_df, self.ideal_df)
        error = func_matcher.calc_leastsq('y1', 'y1')  # 2x vs 5x = different
        self.assertGreater(error, 0.0)

    def test_best_match_finds_correct(self):
        """best_match should pick the ideal function with the smallest error"""
        func_matcher = FunctionMatch(self.train_df, self.ideal_df)
        best_col, error = func_matcher.best_match('y1')
        self.assertEqual(best_col, 'y2')  # train y1=2x should match ideal y2=2x
        self.assertAlmostEqual(error, 0.0)

    def test_match_all_returns_four(self):
        """match_all should return exactly 4 matches (one per training column)"""
        func_matcher = FunctionMatch(self.train_df, self.ideal_df)
        match_result = func_matcher.match_all()
        self.assertEqual(len(match_result), 4)

    def test_match_all_correct_mapping(self):
        """each training column should map to its correct ideal"""
        func_matcher = FunctionMatch(self.train_df, self.ideal_df)
        match_result = func_matcher.match_all()
        self.assertEqual(match_result['y1'], 'y2')   # 2x -> 2x
        self.assertEqual(match_result['y2'], 'y3')   # x^2 -> x^2
        self.assertEqual(match_result['y3'], 'y4')   # sin(x) -> sin(x)
        self.assertEqual(match_result['y4'], 'y5')   # 3x+1 -> 3x+1

    def test_max_deviations_calculated(self):
        """match_all should fill the max_dev_values dict with 4 entries"""
        func_matcher = FunctionMatch(self.train_df, self.ideal_df)
        func_matcher.match_all()
        self.assertEqual(len(func_matcher.max_dev_values), 4)

    def test_incompatible_x_values(self):
        """should raise DataLoadingError if x values are different between train and ideal"""
        invalid_ideal = self.ideal_df.copy()
        invalid_ideal['x'] = invalid_ideal['x'] + 100
        with self.assertRaises(DataLoadingError):
            FunctionMatch(self.train_df, invalid_ideal)

class TestTestClassifier(unittest.TestCase):
    """
    Tests for TestClassifier class (sqrt(2) criterion)
    """

    def setUp(self):
        """creating sample data and run the matcher first to get the matches"""
        self.x_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        train_df = pd.DataFrame({
            'x': self.x_values,
            'y1': self.x_values * 2,
            'y2': self.x_values ** 2,
            'y3': np.sin(self.x_values),
            'y4': self.x_values * 3 + 1
        })
        self.ideal_df = pd.DataFrame({
            'x': self.x_values,
            'y1': self.x_values * 5,
            'y2': self.x_values * 2,
            'y3': self.x_values ** 2,
            'y4': np.sin(self.x_values),
            'y5': self.x_values * 3 + 1
        })
        # running matcher to get matched_functions and max_dev_values
        func_matcher = FunctionMatch(train_df, self.ideal_df)
        self.matched_functions = func_matcher.match_all()
        self.max_dev_values = func_matcher.max_dev_values

    def test_classify_exact_point(self):
        """a point exactly on the ideal function should get mapped with delta=0"""
        test_df = pd.DataFrame({
            'x': [2.0],
            'y': [4.0]  # 2*2 = 4.0, exact match with ideal y2
        })
        classifier = TestClassifier(
            test_df, self.ideal_df,
            self.matched_functions, self.max_dev_values
        )
        result = classifier.classify_point(2.0, 4.0)
        self.assertIsNotNone(result)
        self.assertEqual(result['ideal_func'], 'y2')
        self.assertAlmostEqual(result['delta_y'], 0.0)

    def test_classify_far_point_not_mapped(self):
        """a point really far from everything should not get mapped"""
        test_df = pd.DataFrame({
            'x': [1.0],
            'y': [9999.0]
        })
        classifier = TestClassifier(
            test_df, self.ideal_df,
            self.matched_functions, self.max_dev_values
        )
        result = classifier.classify_point(1.0, 9999.0)
        self.assertIsNone(result)

    def test_classify_all_returns_dataframe(self):
        """classify_all should always return a DataFrame"""
        test_df = pd.DataFrame({
            'x': [1.0, 2.0],
            'y': [2.0, 4.0]
        })
        classifier = TestClassifier(
            test_df, self.ideal_df,
            self.matched_functions, self.max_dev_values
        )
        classification_output = classifier.classify_all()
        self.assertIsInstance(classification_output, pd.DataFrame)

    def test_classify_all_correct_columns(self):
        """results should have the 4 required columns"""
        test_df = pd.DataFrame({
            'x': [1.0],
            'y': [2.0]
        })
        classifier = TestClassifier(
            test_df, self.ideal_df,
            self.matched_functions, self.max_dev_values
        )
        classification_output = classifier.classify_all()
        if not classification_output.empty:
            required_cols = {'x', 'y', 'delta_y', 'ideal_func'}
            self.assertEqual(set(classification_output.columns), required_cols)

    def test_classify_nonexistent_x(self):
        """if the x value doesnt exist in ideal_df it should return None"""
        test_df = pd.DataFrame({
            'x': [999.0],
            'y': [1.0]
        })
        classifier = TestClassifier(
            test_df, self.ideal_df,
            self.matched_functions, self.max_dev_values
        )
        result = classifier.classify_point(999.0, 1.0)
        self.assertIsNone(result)

class TestDatabaseFunctions(unittest.TestCase):
    """
    Tests for database creation and storage functions
    """

    def setUp(self):
        """using a temporary database name so we dont use with the real one"""
        self.test_db = "test_temp.db"

    def tearDown(self):
        """cleaning up the temp database after each test"""
        try:
            if hasattr(self, 'engine'):
                self.engine.dispose()
        except:
            pass
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except PermissionError:
                pass

    def test_create_database(self):
        """create_database should return a working engine"""
        self.engine = create_database(self.test_db)
        self.assertIsNotNone(self.engine)

    def test_save_and_read_from_database(self):
        """data saved should be the same when we read it back"""
        self.engine = create_database(self.test_db)
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        store_db(self.engine, df, "test_table")
        result = pd.read_sql("SELECT * FROM test_table", self.engine)
        self.assertEqual(len(result), 3)
        self.assertListEqual(list(result.columns), ['x', 'y'])

    def test_load_csv_file_not_found(self):
        """load_csv should raise DataLoadingError if the file dont exist"""
        with self.assertRaises(DataLoadingError):
            load_csv("nonexistent_file.csv")

if __name__ == "__main__":
    unittest.main()
