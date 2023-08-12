# Doping_Outlier_Tester
Tool to modify a given dataset with known anomalies in order to test outlier detectors

## Motivation
This is a simple tool that can provide one method to test outlier detectors. Given a dataset, in the form of a pandas dataframe, the tool randomly modifies a small set of values within the dataframe. In most cases, this will replace a common value (by virtue of the cells being selected randomly), with another value, where the new value will tend to be less common for the column. The new value may be either an existing value or a new value for the column. Where the new value is a previously existing value, it may be a relatively common value, but even where the original value is replaced with another common value, so long as the value in different (in the case of categorical columns) or significantly different (in the case of numeric columns), the new value will typically form unusual combinations with the other features in the dataset, where there are correlations among the features. 

The tool does not attempt to quantify how unusual the new values are, as this presumes some objective measure of the outlierness of any given row, which is not possible, and it is the role of outlier detectors to estimate this. It does, however, distinguish between cases where the new value did exist previously in the column and where it is completely novel for the column.

The tool has a single API, transform(), which takes a pandas dataframe, modifies a small set of cells and returns the modified dataframe. The dataframe has one additional column added, 'OUTLIER SCORE', which is an estimate of how unusual each row is relative to its state prior to transforming the dataset. This is based on: 1) how many cells within the row were modified; and 2) if the new values were novel to the column or within the set of previous values for that column. 

Although it is not possible to estimate how anomalous the modified rows are, intuitively they will be, on average, more unusual than the unmodified rows, and the more columns modfied within a row, the more anomalous it will be relative to its original state. Experiments describe below confirm this.

## Algorithm
A small set of rows should be modified, as modifiying a significant number of rows can change the marginal and joint distributions of values in the dataset, changing the sense of what constitutes an outlier. The idea is to insert known values into a dataset such that we can expect that in most cases the new values will be unusual in some sense, though we do not know how anomalous the modified rows are

## Example
More involved examples are provided in the [example notebook](https://github.com/Brett-Kennedy/Doping_Outlier_Tester/blob/main/examples/Simple%20Example.ipynb), but the tool is quite simple, uses a single API and in most cases the default parameters will be sufficient, though may be modified where you wish to qualify better where different detectors tend to have stronger or weaker performance.

```python
import pandas as pd
from sklearn.datasets import fetch_openml
from outliers_test import DopingOutliersTest

data = fetch_openml('breast-w', version=1)
df = pd.DataFrame(data.data, columns=data.feature_names)

data_modifier = DopingOutliersTest()
df_modified = data_modifier.transform(df, verbose=True)
```

## Parameters

**df**: dataframe

&nbsp;&nbsp;The original dataframe. This dataframe itself will be unmodified, but another, modified copy of the dataframe will be returned. 

&NewLine;  

**num_rows_to_modify**: int 
            
&nbsp;&nbsp;The number of rows that are modified


**min_cols_per_modification**:  int

&nbsp;&nbsp;Each modified row will have at least min_cols_per_modification values modified.

**max_cols_per_modification**: int 

&nbsp;&nbsp;Each modified row will have at most max_cols_per_modification values modified.

**allow_new_categorical_values**: bool

&nbsp;&nbsp;If True, categorical columns that are modified will in some cases be given new values unique to the column. If False, categorical columns that are modified will be given new values that already existed elsewhere in the column. 

**allow_new_numeric_values**: bool

&nbsp;&nbsp;If True, numeric values that are modified will in some cases be given new values that are larger than those already in the column. If False, numeric values that are modified will be given new values that are within the range of the existing values in the column. 

**random_state**: int 

&nbsp;&nbsp; May be set to ensure consistent results. If set to -1, no seed will be used. 

**verbose**: bool

&nbsp;&nbsp;If True, some messages will be displayed to provide an indication of the progress of the detector.

## Return Value
The returned dataframe will be almost identical to the original dataframe, with a small number of rows
will some modified values, and an additional column 'OUTLIER SCORE', which estimates how anomalous the 
modified rows are relative to their original state. All unmodified rows will have a zero in this column.

## Demo Notebook
The [example notebook](https://github.com/Brett-Kennedy/Doping_Outlier_Tester/blob/main/examples/Simple%20Example.ipynb) provides an example using IsolationForest (IF), which is known to be a strong outlier detector. 

The Doping Outliers Test tool cannot estimate how unusual any row is, only how unusual it likely is compared to its state before the doping process. As such, to evaluate how well IF does in identifying the modified rows, we get the IF score of each row both before and after the doping process and examine the difference in IF scores. The gains in outlier scores for the modified rows should be higher than for the unmodified rows, which is, in fact, the case. The IF scores (once cleaned) correlate very closely with the estimated scores produced by the Doping Outlier Tester tool.

The notebook also displays the specific gain in IF score and estimate outlier score for the 10 rows modified as well as 10 other rows. We see both
are zero and non-zero in the same rows. This is not strictly true for all datasets, but tends to be the case.

