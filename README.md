# Doping_Outlier_Tester
Tool to modify a given dataset with known anomalies in order to test outlier detectors

## Motivation
This is a simple tool that can provide one method to test outlier detectors. Given a dataset, in the form of a pandas dataframe, the tool randomly modifies a small set of values within the dataframe. In most cases, this will replace a common value (by virtue of the cells being selected randomly), with another value, where the new value will tend to be less common for the column. It may either be an existing value, or a new value for the column. Even where the original value is replaced with another common value, so long as the value in different (in the case of categorical columns) or significantly different (in the case of numeric columns), the new value will typically form unusual combinations with the other features in the dataset, where there are correlations among the features. 

The tool does not attempt to quantify how unusual the new values are, as this presumes some objects measure of the outlierness of any given row, which is not possible. It does, however, distinguish between cases where the new value did exist previously in the column and where it is completely novel.

## Algorithm


## Example

The tool has a single API, transform(), which takes a pandas dataframe, modifies a small set of cells and returns the modified dataframe. The dataframe has one additional column added, 'OUTLIER SCORE', which is an estimate of how unusual each row is relative to its state prior to transforming the dataset. This is based on: 1) how many cells within the row were modified; and 2) if the new values were novel to the column or within the set of previous values for that column. 

## Parameters

```
df: dataframe
            The original dataframe. This dataframe itself will be unmodified, but a  

num_rows_to_modify: int 
            The number of rows that are modified

min_cols_per_modification: 
            Each modified row will have at least min_cols_per_modification values modified.

max_cols_per_modification: int 
            Each modified row will have at most max_cols_per_modification values modified.

allow_new_categorical_values: 
            If True, categorical columns that are modified will in some cases be given new values unique to the column.
            If False, categorical columns that are modified will be given new values that already existed elsewhere
            in the column. 

allow_new_numeric_values:
             If True, numeric values that are modified will in some cases be given new values that are larger than
             those already in the column. If False, numeric values that are modified will be given new values that
             are within the range of the existing values in the column. 

random_state: int 
            May be set to ensure consistent results. If set to -1, no seed will be used. 
verbose: 
            If True, some messages will be displayed to provide an indication of the progress of the detector.

## Return Value
The returned dataframe will be almost identical to the original dataframe, with a small number of rows
will some modified values, and an additional column 'OUTLIER SCORE', which estimates how anomalous the 
modified rows are relative to their original state. All unmodified rows will have a zero in this column.


```
## Demo Notebook


