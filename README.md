# DopingOutlierTester
Tool to modify a given dataset with known, random anomalies in order to test outlier detectors.

## Motivation
It is difficult to estimate how well different outlier detectors perform on any given type of data given the lack of ground truth. A number of techniques may be used including:

- Taking datasets normally used for classification, where there is one clear minority class, and taking this class to be the outliers. This method has a number of flaws, including the issue that the function mapping the feature values to the target column may use only a small number of features (and may use these in a complex way), while an outlier detector treats each feature equally. 
- Agreement with other outlier detectors
- Manual inspection. This can work in some cases, but is error-prone and intractable for large datasets.
- Testing with synthetic data. Here data may be generated having known distributions, with known exceptions to these distributions. This can work well with low dimensionalities, but synthetic data may inadvertently contain any number of unknown anomalies, as it is not feasible to specify the joint distributions for every subset of columns. 

These techniques can be useful, though do have their limitations. 

This is a simple tool that can provide *doping*, another method to test outlier detectors: starting with a real or synthetic dataset, a small number of random modifications are made, keeping track of the modified rows. Given a dataset, in the form of a pandas dataframe, the tool randomly modifies a small set of values within the dataframe. In most cases, this will replace a common value (by virtue of the cells being selected randomly), with another value, where the new value will tend to be less common for the column. The new value may be either an existing value or a new value for the column. Where the new value is a previously-existing value, it may, in some cases, be a relatively common value, but even where the original value is replaced with another common value, so long as the value is different (in the case of categorical columns) or significantly different (in the case of numeric columns), the new value will typically form unusual combinations with the other features in the dataset, where there are correlations or some relations among the features. The tool ensures the updated values are different.

The tool does not attempt to quantify how unusual the new values are, as this presumes some objective measure of the outlierness of any given row, which is not possible, and it is the role of outlier detectors to estimate this. It does, however, consider the number of cells modified, and distinguishes between cases where the new value did exist previously in the column and where it is completely novel for the column.

The tool has a single API, transform(), which takes a pandas dataframe, modifies a small set of cells and returns the modified dataframe, as well as an array, which is an estimate of how unusual each row is relative to its state prior to transforming the dataset. This is based on: 1) how many cells within the row were modified; and 2) if the new values were novel to the column or within the set of previous values for that column. This will have zero values for the unmodified rows and non-zero values for the modified rows. 

Although it is not possible to estimate how anomalous the modified rows are, intuitively they will be, on average, more unusual than the unmodified rows, and the more columns modfied within a row, the more anomalous it will be relative to its original state. More relevantly, the modified rows will be more anomalous than they previously were. Experiments describe below confirm this.

## Algorithm
A small set of rows should be modified, as modifiying a significant number of rows can change the marginal and joint distributions of values in the dataset, changing the sense of what constitutes an outlier. The idea is to insert a small set of modified (doped) values in known locations into a dataset such that we can expect that in most cases the modified rows will be unusual in some sense, even though we do not know specifically how anomalous the modified rows are. 

The tool selects a random set of rows to be modified, and for each a random set of columns. For categorical columns, it either creates a new value for the column (if specified to allow this), or selects a random value from the column. For numeric values, it either creates a value outside the previous range of values for the column (if specified to allow this), or selects a value from within the existing range. 

## Example
A more involved example is provided in the [example notebook](https://github.com/Brett-Kennedy/Doping_Outlier_Tester/blob/main/examples/Simple%20Example.ipynb), but the tool is quite simple, uses a single API and in most cases the default parameters will be sufficient, though may be modified where you wish to qualify better where different detectors tend to have stronger or weaker performance.

```python
import pandas as pd
from sklearn.datasets import fetch_openml
from outliers_test import DopingOutliersTest

data = fetch_openml('breast-w', version=1)
df = pd.DataFrame(data.data, columns=data.feature_names)

data_modifier = DopingOutliersTest()
df_modified, scores_arr = data_modifier.transform(df)
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
will some modified values. An array, which estimates how anomalous the 
modified rows are relative to their original state. This will have an element for each row in the dataset. 
All unmodified rows will have a zero in this array.

## Demo Notebooks

**Simple Example Notebook**

The [example notebook](https://github.com/Brett-Kennedy/Doping_Outlier_Tester/blob/main/examples/Simple%20Example.ipynb) provides an example using IsolationForest (IF), which is known to be a strong outlier detector. 

The Doping Outliers Test tool cannot estimate how unusual any row is, only how unusual it likely is compared to its state before the doping process. As such, to evaluate how well IF does in identifying the modified rows, we get the IF score of each row both before and after the doping process and examine the difference in IF scores. The gains in outlier scores for the modified rows should be higher than for the unmodified rows, which is, in fact, the case. The IF scores (once cleaned) correlate very closely with the estimated scores produced by the Doping Outlier Tester tool.

The notebook also displays the specific gain in IF score and estimate outlier score for the 10 rows modified as well as 10 other rows. We see both
are zero and non-zero in the same rows. This is not strictly true for all datasets, but tends to be the case.

**Test Doping OpenML Notebook**

The [OpenML test notebook](https://github.com/Brett-Kennedy/DopingOutlierTester/blob/main/examples/Test_Doping_OpenML_IF.ipynb) provides a more thorough test of the tool, ensuring that, given a dataset, both IsolationForest (IF) and Local Outlier Factor (LOF) are able to detect modifications in a dataset reliably. This was tested on a large number of datasets from OpenML selected randomly. 

In this test, each dataset was modified by DopingOutliersTest using default parameters, which modifies ten rows. With each dataset, we get the outlier scores on both the original and modified forms of the dataset using both IF and LOF. We then check the gains in scores. We evaluate the detectors in two key ways.

1) We take the top 10 scores for each detector on each dataset. In this case, we know the doping process modified 10 rows, but this information will not typically be available. As it can be difficult with outlier detectors to determine the best cut-off, this test is included to simulate where there is a reasonable guess as to the number of outliers. It demonstrates that the detectors rank the scores well such that those modified by the doping process tend to have the highest scores, even if the ideal cutoff remains elusive. 

2) We use a common technique in outlier detection to determine a cutoff, testing the set of outlier scores for extreme values, and taking any unusually high scores as outliers. For this we calculate the interquartile range, and use a coefficient of 2.2, which is standard for IQR tests, on the IF scores. The LOF scores, however, are more dispersed, and a coefficient of 22.0 was used instead. 

This demonstrates that IF and LOF are both generally able to give higher scores to modified rows after the doping process than before, and not give higher scores to unmodified rows. 

Results:
```
Average IF Jaccard Similarity to top 10:  0.69
Average LOF Jaccard Similarity to top 10: 0.64
Average IF Jaccard Similarity using IQR:  0.56
Average LOF Jaccard Similarity using IQR: 0.32
```
Although explicitily using the top ten scores performs better, all show a very high degree of similarity between the top scores of the detectors and the truly modified rows. Even 0.32 is a quite decent Jaccard similarity, and this can be improved with other methods to determine a cutoff for a binary outlier flag using LOF.

We may conclude that the DopingOutliersTest is able to modify datasets in a manner which outlier detectors can generally identify. However, each outlier detector uses its own algorithm, which allows each to identify different types of outliers. Many of the changes can be detected better by IF, with others better by LOF, but on average both performed quite well. 
