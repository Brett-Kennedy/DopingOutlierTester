import pandas as pd
import numpy as np
import random
from pandas.api.types import is_numeric_dtype


class DopingOutliersTest:
    def __init__(self):
        self.df = None

    def transform(self,
                  df,
                  num_rows_to_modify=10,
                  min_cols_per_modification=1,
                  max_cols_per_modification=-1,
                  allow_new_categorical_values=True,
                  allow_new_numeric_values=True,
                  random_state=-1,
                  verbose=False
                  ):

        """

        :param df: dataframe
            The original dataframe. This dataframe itself will be unmodified, but another, modified copy of the
            dataframe will be returned.
        :param num_rows_to_modify: int
            The number of rows that are modified
        :param min_cols_per_modification: int
            Each modified row will have at least min_cols_per_modification values modified.
        :param max_cols_per_modification: int
            Each modified row will have at most max_cols_per_modification values modified.
        :param allow_new_categorical_values: bool
            If True, categorical columns that are modified will in some cases be given new values unique to the column.
            If False, categorical columns that are modified will be given new values that already existed elsewhere
            in the column.
        :param allow_new_numeric_values: bool
             If True, numeric values that are modified will in some cases be given new values that are larger than
             those already in the column. If False, numeric values that are modified will be given new values that
             are within the range of the existing values in the column.
        :param random_state: int
            May be set to ensure consistent results. If set to -1, no seed will be used.
        :param verbose: bool
            If True, some messages will be displayed to provide an indication of the progress of the detector.
        :return: dataframe
            The returned dataframe will be almost identical to the original dataframe, with a small number of rows
            will some modified values, and an additional column 'OUTLIER SCORE', which estimates how anomalous the
            modified rows are relative to their original state. All unmodified rows will have a zero in this column.
        """

        if random_state >= 0:
            np.random.seed(random_state)
            random.seed(random_state)

        self.df = df.copy()
        num_rows = len(df)
        num_cols = len(df.columns)

        # Get the type of each column. We handle modifying categorical vs numeric columns differently
        col_types_arr = self.__get_col_types_arr()

        if max_cols_per_modification < 0:
            max_cols_per_modification = num_cols

        # Ensure the parameters have valid values
        if num_rows_to_modify < 1:
            print("num_rows_to_modify must be at least 1")
            return
        if num_rows_to_modify > num_rows:
            print("num_rows_to_modify must be at most the number of rows in the dataframe")
            return
        if min_cols_per_modification < 1:
            print("min_cols_per_modification must be at least 1")
            return
        if min_cols_per_modification > num_cols:
            print("min_cols_per_modification must be at most the number of columns in the dataframe")
            return
        if max_cols_per_modification > num_cols:
            print("max_cols_per_modification must be at most the number of columns in the dataframe")
            return

        # Initialize outlier scores. Most rows will have score 0. Those that are modified will have a score
        # representing the number of values that changed and whether they changed to an previously unseen value or not.
        outlier_scores = [0] * num_rows

        # Pick the set of rows to modify
        modified_rows = np.random.choice(list(range(num_rows)), num_rows_to_modify)

        # Loop through each row that's modified
        new_vals_counter = 1
        for row_count, row_idx in enumerate(modified_rows):

            # Determine the columns modified for this row
            num_cols_modified = -1
            while num_cols_modified < min_cols_per_modification or num_cols_modified > max_cols_per_modification:
                num_cols_modified = int(abs(np.random.laplace(1.0, 10)))
            modified_cols = np.random.choice(df.columns, num_cols_modified)
            if verbose:
                print(f"Modifying row {row_count} of {num_rows_to_modify}. Modifying {num_cols_modified} columns")

            # Loop through each column and modify it
            for col_name in modified_cols:
                col_idx = df.columns.tolist().index(col_name)
                new_val = None
                create_new_value = False
                if col_types_arr[col_idx] == 'C':
                    if allow_new_categorical_values:
                        p = np.random.random()
                        if p < 0.5:
                            create_new_value = True
                    if create_new_value:
                        new_val = "NEW VALUE" + str(new_vals_counter)
                        new_vals_counter += 1
                    else:
                        unique_vals = df[col_name].unique()
                        new_val = np.random.choice(unique_vals)
                else:  # Numeric
                    col_min = self.df[col_name].min()
                    col_max = self.df[col_name].max()
                    if allow_new_numeric_values:
                        p = np.random.random()
                        if p < 0.5:
                            create_new_value = True
                    if create_new_value:
                        new_val = col_max + (np.random.random() * (col_max - col_min))
                    else:
                        new_val = col_min + (np.random.random() * (col_max - col_min))

                outlier_scores[row_idx] += 2 if create_new_value else 1
                self.df.loc[row_idx, col_name] = new_val

        # Add the approximate outlier scores to each row
        self.df['OUTLIER SCORE'] = outlier_scores

        return self.df

    def __get_col_types_arr(self):
        """
        Create an array representing each column of the data, with each coded as 'C' (categorical) or 'N' (numeric).
        """

        col_types_arr = ['N'] * len(self.df.columns)

        for col_idx, col_name in enumerate(self.df.columns):
            is_numeric = self.__get_is_numeric(col_name)
            if not is_numeric:
                col_types_arr[col_idx] = 'C'

        # Ensure any numeric columns are stored in integer or float format and categorical as strings
        for col_idx, col_name in enumerate(self.df.columns):
            if col_types_arr[col_idx] == 'C':
                self.df[col_name] = self.df[col_name].astype(str)
            if col_types_arr[col_idx] == 'N':
                if not self.df[col_name].dtype in [int, np.int64]:
                    self.df[col_name] = self.df[col_name].astype(float)

        return col_types_arr

    def __get_is_numeric(self, col_name):
        is_numeric = is_numeric_dtype((self.df[col_name])) or \
                     (self.df[col_name]
                      .astype(str).str.replace('-', '', regex=False)
                      .str.replace('.', '', regex=False)
                      .str.isdigit()
                      .tolist()
                      .count(False) == 0)
        return is_numeric
