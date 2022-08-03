---
template: lesson.html
title: Pandas for Machine Learning
description: Data manipulation using the Pandas library.
keywords: pandas, exploratory data analysis, eda, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/foundations.png
repository: https://github.com/GokuMohandas/Made-With-ML
notebook: https://github.com/GokuMohandas/Made-With-ML/blob/main/notebooks/04_Pandas.ipynb
---

{% include "styles/lesson.md" %}

## Set up

First we'll import the NumPy and Pandas libraries and set seeds for reproducibility. We'll also download the dataset we'll be working with to disk.
```python linenums="1"
import numpy as np
import pandas as pd
```
```python linenums="1"
# Set seed for reproducibility
np.random.seed(seed=1234)
```
<pre class="output"></pre>

## Load data
We're going to work with the [Titanic dataset](https://www.kaggle.com/c/titanic/data){:target="_blank"} which has data on the people who embarked the RMS Titanic in 1912 and whether they survived the expedition or not. It's a very common and rich dataset which makes it very apt for exploratory data analysis with Pandas.

Let's load the data from the CSV file into a Pandas dataframe. The `header=0` signifies that the first row (0th index) is a header row which contains the names of each column in our dataset.

```python linenums="1"
# Read from CSV to Pandas DataFrame
url = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/titanic.csv"
df = pd.read_csv(url, header=0)
```
```python linenums="1"
# First few items
df.head(3)
```
<div class="output_subarea output_html rendered_html ai-center-all"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pclass</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>embarked</th>
      <th>survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Allen, Miss. Elisabeth Walton</td>
      <td>female</td>
      <td>29.0000</td>
      <td>0</td>
      <td>0</td>
      <td>24160</td>
      <td>211.3375</td>
      <td>B5</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Allison, Master. Hudson Trevor</td>
      <td>male</td>
      <td>0.9167</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Allison, Miss. Helen Loraine</td>
      <td>female</td>
      <td>2.0000</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div></div>

These are the different features:

- `class`: class of travel
- `name`: full name of the passenger
- `sex`: gender
- `age`: numerical age
- `sibsp`: # of siblings/spouse aboard
- `parch`: number of parents/child aboard
- `ticket`: ticket number
- `fare`: cost of the ticket
- `cabin`: location of room
- `embarked`: port that the passenger embarked at
- `survived`: survival metric (0 - died, 1 - survived)


## Exploratory data analysis (EDA)
Now that we loaded our data, we're ready to start exploring it to find interesting information.

> Be sure to check out our entire lesson focused on [EDA](https://madewithml.com/courses/mlops/exploratory-data-analysis/){:target="_blank"} in our [MLOps](https://madewithml.com/courses/mlops/){:target="_blank"} course.

```python linenums="1"
import matplotlib.pyplot as plt
```
<pre class="output"></pre>
We can use `.describe()` to extract some standard details about our numerical features.

```python linenums="1"
# Describe features
df.describe()
```
<div class="output_subarea output_html rendered_html ai-center-all"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pclass</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1309.000000</td>
      <td>1046.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1308.000000</td>
      <td>1309.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.294882</td>
      <td>29.881135</td>
      <td>0.498854</td>
      <td>0.385027</td>
      <td>33.295479</td>
      <td>0.381971</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.837836</td>
      <td>14.413500</td>
      <td>1.041658</td>
      <td>0.865560</td>
      <td>51.758668</td>
      <td>0.486055</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.166700</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.895800</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.275000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>512.329200</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div></div>

```python linenums="1"
# Correlation matrix
plt.matshow(df.corr())
continuous_features = df.describe().columns
plt.xticks(range(len(continuous_features)), continuous_features, rotation="45")
plt.yticks(range(len(continuous_features)), continuous_features, rotation="45")
plt.colorbar()
plt.show()
```

<div class="ai-center-all">
    <img src="/static/images/foundations/pandas/correlation.png" width="350" alt="correlation">
</div>

We can also use `.hist()` to view the histogram of values for each feature.
```python linenums="1"
# Histograms
df["age"].hist()
```

<div class="ai-center-all">
    <img src="/static/images/foundations/pandas/histogram.png" width="350" alt="histogram">
</div>

```python linenums="1"
# Unique values
df["embarked"].unique()
```
<pre class="output">
array(['S', 'C', nan, 'Q'], dtype=object)
</pre>



## Filtering
We can filter our data by features and even by specific values (or value ranges) within specific features.
```python linenums="1"
# Selecting data by feature
df["name"].head()
```
<pre class="output">
0                      Allen, Miss. Elisabeth Walton
1                     Allison, Master. Hudson Trevor
2                       Allison, Miss. Helen Loraine
3               Allison, Mr. Hudson Joshua Creighton
4    Allison, Mrs. Hudson J C (Bessie Waldo Daniels)
Name: name, dtype: object
</pre>
```python linenums="1"
# Filtering
df[df["sex"]=="female"].head() # only the female data appear
```

<div class="output_subarea output_html rendered_html ai-center-all"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pclass</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>embarked</th>
      <th>survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Allen, Miss. Elisabeth Walton</td>
      <td>female</td>
      <td>29.0</td>
      <td>0</td>
      <td>0</td>
      <td>24160</td>
      <td>211.3375</td>
      <td>B5</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Allison, Miss. Helen Loraine</td>
      <td>female</td>
      <td>2.0</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td>
      <td>female</td>
      <td>25.0</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>Andrews, Miss. Kornelia Theodosia</td>
      <td>female</td>
      <td>63.0</td>
      <td>1</td>
      <td>0</td>
      <td>13502</td>
      <td>77.9583</td>
      <td>D7</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>Appleton, Mrs. Edward Dale (Charlotte Lamson)</td>
      <td>female</td>
      <td>53.0</td>
      <td>2</td>
      <td>0</td>
      <td>11769</td>
      <td>51.4792</td>
      <td>C101</td>
      <td>S</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div></div>


## Sorting

We can also sort our features in ascending or descending order.
```python linenums="1"
# Sorting
df.sort_values("age", ascending=False).head()
```
<div class="output_subarea output_html rendered_html ai-center-all"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pclass</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>embarked</th>
      <th>survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>Barkworth, Mr. Algernon Henry Wilson</td>
      <td>male</td>
      <td>80.0</td>
      <td>0</td>
      <td>0</td>
      <td>27042</td>
      <td>30.0000</td>
      <td>A23</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>61</th>
      <td>1</td>
      <td>Cavendish, Mrs. Tyrell William (Julia Florence...</td>
      <td>female</td>
      <td>76.0</td>
      <td>1</td>
      <td>0</td>
      <td>19877</td>
      <td>78.8500</td>
      <td>C46</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1235</th>
      <td>3</td>
      <td>Svensson, Mr. Johan</td>
      <td>male</td>
      <td>74.0</td>
      <td>0</td>
      <td>0</td>
      <td>347060</td>
      <td>7.7750</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>135</th>
      <td>1</td>
      <td>Goldschmidt, Mr. George B</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17754</td>
      <td>34.6542</td>
      <td>A5</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>Artagaveytia, Mr. Ramon</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17609</td>
      <td>49.5042</td>
      <td>NaN</td>
      <td>C</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div></div>


## Grouping

We can also get statistics across our features for certain groups. Here we wan to see the average of our continuous features based on whether the passenger survived or not.
```python linenums="1"
# Grouping
survived_group = df.groupby("survived")
survived_group.mean()
```
<div class="output_subarea output_html rendered_html ai-center-all"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>survived</th>
      <th>pclass</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.500618</td>
      <td>30.545369</td>
      <td>0.521632</td>
      <td>0.328801</td>
      <td>23.353831</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.962000</td>
      <td>28.918228</td>
      <td>0.462000</td>
      <td>0.476000</td>
      <td>49.361184</td>
    </tr>
  </tbody>
</table>
</div></div>


## Indexing

We can use `iloc` to get rows or columns at particular positions in the dataframe.
```python linenums="1"
# Selecting row 0
df.iloc[0, :]
```
<pre class="output">
pclass                                  1
name        Allen, Miss. Elisabeth Walton
sex                                female
age                                    29
sibsp                                   0
parch                                   0
ticket                              24160
fare                              211.338
cabin                                  B5
embarked                                S
survived                                1
Name: 0, dtype: object
</pre>
```python linenums="1"
# Selecting a specific value
df.iloc[0, 1]
```
<pre class="output">
'Allen, Miss. Elisabeth Walton'
</pre>

## Preprocessing
After exploring, we can clean and preprocess our dataset.

> Be sure to check out our entire lesson focused on [preprocessing](https://madewithml.com/courses/mlops/preprocessing/){:target="_blank"} in our [MLOps](https://madewithml.com/courses/mlops/){:target="_blank"} course.

```python linenums="1"
# Rows with at least one NaN value
df[pd.isnull(df).any(axis=1)].head()
```
<div class="output_subarea output_html rendered_html ai-center-all"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pclass</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>embarked</th>
      <th>survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>Artagaveytia, Mr. Ramon</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17609</td>
      <td>49.5042</td>
      <td>NaN</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>Barber, Miss. Ellen "Nellie"</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>19877</td>
      <td>78.8500</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>Baumann, Mr. John D</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17318</td>
      <td>25.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>Bidois, Miss. Rosalie</td>
      <td>female</td>
      <td>42.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17757</td>
      <td>227.5250</td>
      <td>NaN</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>Birnbaum, Mr. Jakob</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>13905</td>
      <td>26.0000</td>
      <td>NaN</td>
      <td>C</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div></div>

```python linenums="1"
# Drop rows with Nan values
df = df.dropna() # removes rows with any NaN values
df = df.reset_index() # reset's row indexes in case any rows were dropped
df.head()
```
<div class="output_subarea output_html rendered_html ai-center-all"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>pclass</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>embarked</th>
      <th>survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>Allen, Miss. Elisabeth Walton</td>
      <td>female</td>
      <td>29.0000</td>
      <td>0</td>
      <td>0</td>
      <td>24160</td>
      <td>211.3375</td>
      <td>B5</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Allison, Master. Hudson Trevor</td>
      <td>male</td>
      <td>0.9167</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>Allison, Miss. Helen Loraine</td>
      <td>female</td>
      <td>2.0000</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>Allison, Mr. Hudson Joshua Creighton</td>
      <td>male</td>
      <td>30.0000</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td>
      <td>female</td>
      <td>25.0000</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div></div>

```python linenums="1"
# Dropping multiple columns
df = df.drop(["name", "cabin", "ticket"], axis=1) # we won't use text features for our initial basic models
df.head()
```
<div class="output_subarea output_html rendered_html ai-center-all"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>29.0000</td>
      <td>0</td>
      <td>0</td>
      <td>211.3375</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>0.9167</td>
      <td>1</td>
      <td>2</td>
      <td>151.5500</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>female</td>
      <td>2.0000</td>
      <td>1</td>
      <td>2</td>
      <td>151.5500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>male</td>
      <td>30.0000</td>
      <td>1</td>
      <td>2</td>
      <td>151.5500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>female</td>
      <td>25.0000</td>
      <td>1</td>
      <td>2</td>
      <td>151.5500</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div></div>

```python linenums="1"
# Map feature values
df["sex"] = df["sex"].map( {"female": 0, "male": 1} ).astype(int)
df["embarked"] = df["embarked"].dropna().map( {"S":0, "C":1, "Q":2} ).astype(int)
df.head()
```
<div class="output_subarea output_html rendered_html ai-center-all"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>29.0000</td>
      <td>0</td>
      <td>0</td>
      <td>211.3375</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.9167</td>
      <td>1</td>
      <td>2</td>
      <td>151.5500</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2.0000</td>
      <td>1</td>
      <td>2</td>
      <td>151.5500</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>30.0000</td>
      <td>1</td>
      <td>2</td>
      <td>151.5500</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>25.0000</td>
      <td>1</td>
      <td>2</td>
      <td>151.5500</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div></div>


## Feature engineering
We're now going to use feature engineering to create a column called `family_size`. We'll first define a function called `get_family_size` that will determine the family size using the number of parents and siblings.
```python linenums="1"
# Lambda expressions to create new features
def get_family_size(sibsp, parch):
    family_size = sibsp + parch
    return family_size
```
Once we define the function, we can use `lambda` to `apply` that function on each row (using the numbers of siblings and parents in each row to determine the family size for each row).
```python linenums="1"
df["family_size"] = df[["sibsp", "parch"]].apply(lambda x: get_family_size(x["sibsp"], x["parch"]), axis=1)
df.head()
```

<div class="output_subarea output_html rendered_html ai-center-all"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>survived</th>
      <th>family_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>29.0000</td>
      <td>0</td>
      <td>0</td>
      <td>211.3375</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.9167</td>
      <td>1</td>
      <td>2</td>
      <td>151.5500</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2.0000</td>
      <td>1</td>
      <td>2</td>
      <td>151.5500</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>30.0000</td>
      <td>1</td>
      <td>2</td>
      <td>151.5500</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>25.0000</td>
      <td>1</td>
      <td>2</td>
      <td>151.5500</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div></div>

```python linenums="1"
# Reorganize headers
df = df[["pclass", "sex", "age", "sibsp", "parch", "family_size", "fare", '"mbarked", "survived"]]
df.head()
```
<div class="output_subarea output_html rendered_html ai-center-all"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>family_size</th>
      <th>fare</th>
      <th>embarked</th>
      <th>survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>29.0000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>211.3375</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0.9167</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>151.5500</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>2.0000</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>151.5500</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>30.0000</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>151.5500</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>25.0000</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>151.5500</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div></div>

!!! tip
    Feature engineering can be done in collaboration with domain experts that can guide us on what features to engineer and use.

## Save data
Finally, let's save our preprocessed data into a new CSV file to use later.
```python linenums="1"
# Saving dataframe to CSV
df.to_csv("processed_titanic.csv", index=False)
```
```python linenums="1"
# See the saved file
!ls -l
```
<pre class="output">
total 96
-rw-r--r-- 1 root root  6975 Dec  3 17:36 processed_titanic.csv
drwxr-xr-x 1 root root  4096 Nov 21 16:30 sample_data
-rw-r--r-- 1 root root 85153 Dec  3 17:36 titanic.csv
</pre>

## Scaling
When working with very large datasets, our Pandas DataFrames can become very large and it can be very slow or impossible to operate on them. This is where packages that can distribute workloads or run on more efficient hardware can come in handy.

- [Dask](https://dask.org/){;target="_blank"}: parallel computing to scale packages like Numpy, Pandas and scikit-learn on one/multiple machines.
- [cuDF](https://github.com/rapidsai/cudf){:target="_blank"}: efficient dataframe loading and computation on a GPU.

And, of course, we can combine these together ([Dask-cuDF](https://github.com/rapidsai/cudf/tree/main/python/dask_cudf){:target="_blank"}) to operate on partitions of a dataframe on the GPU.

<!-- Citation -->
{% include "cite.md" %}