---
template: lesson.html
title: Pandas for Machine Learning
description: Data manipulation using the Pandas library.
keywords: pandas, exploratory data analysis, eda, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/basics.png
repository: https://github.com/GokuMohandas/MadeWithML
notebook: https://colab.research.google.com/github/GokuMohandas/MadeWithML/blob/main/notebooks/04_Pandas.ipynb
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
url = "https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/titanic.csv"
df = pd.read_csv(url, header=0)
```
```python linenums="1"
# First few items
df.head(3)
```
<div class="output_subarea output_html rendered_html"><div>
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
- `emarked`: port that the passenger embarked at (C - Cherbourg, S - Southampton, Q - Queenstown)
- `survived`: survial metric (0 - died, 1 - survived)


## Exploratory data analysis (EDA)
Now that we loaded our data, we're ready to start exploring it to find interesting information.

!!! note
    Be sure to check out our entire lesson focused on [EDA](https://madewithml.com/courses/mlops/exploratory-data-analysis/){:target="_blank"} in our [mlops](https://madewithml.com/courses/mlops/){:target="_blank"} course.

```python linenums="1"
import matplotlib.pyplot as plt
```
<pre class="output"></pre>
We can use `.describe()` to extract some standard details about our numerical features.

```python linenums="1"
# Describe features
df.describe()
```
<div class="output_subarea output_html rendered_html"><div>
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
plt.xticks(range(len(continuous_features)), continuous_features, rotation='45')
plt.yticks(range(len(continuous_features)), continuous_features, rotation='45')
plt.colorbar()
plt.show()
```
<pre class="output"></pre>
<div class="ai-center-all">
    <img width="300" src="data:image/png;base64,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%0AAElFTkSuQmCC">
</div>

We can also use `.hist()` to view the histrogram of values for each feature.
```python linenums="1"
# Histograms
df['age'].hist()
```
<pre class="output"></pre>
<div class="ai-center-all">
    <img width="300" src="data:image/png;base64,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%0AS1KB/g+tXaXiUCPD1wAAAABJRU5ErkJggg==">
</div>

```python linenums="1"
# Unique values
df['embarked'].unique()
```
<pre class="output">
array(['S', 'C', nan, 'Q'], dtype=object)
</pre>



## Filtering
We can filter our data by features and even by specific values (or value ranges) within specific features.
```python linenums="1"
# Selecting data by feature
df['name'].head()
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
df[df['sex']=='female'].head() # only the female data appear
```

<div class="output_subarea output_html rendered_html"><div>
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
df.sort_values('age', ascending=False).head()
```
<div class="output_subarea output_html rendered_html"><div>
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
survived_group = df.groupby('survived')
survived_group.mean()
```
<div class="output_subarea output_html rendered_html"><div>
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

!!! note
    Be sure to check out our entire lesson focused on [preprocessing](https://madewithml.com/courses/mlops/preprocessing/){:target="_blank"} in our [mlops](https://madewithml.com/courses/mlops/){:target="_blank"} course.

```python linenums="1"
# Rows with at least one NaN value
df[pd.isnull(df).any(axis=1)].head()
```
<div class="output_subarea output_html rendered_html"><div>
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
<div class="output_subarea output_html rendered_html"><div>
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
df = df.drop(['name', 'cabin', 'ticket'], axis=1) # we won't use text features for our initial basic models
df.head()
```
<div class="output_subarea output_html rendered_html"><div>
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
df['sex'] = df['sex'].map( {'female': 0, 'male': 1} ).astype(int)
df['embarked'] = df['embarked'].dropna().map( {'S':0, 'C':1, 'Q':2} ).astype(int)
df.head()
```
<div class="output_subarea output_html rendered_html"><div>
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

<div class="output_subarea output_html rendered_html"><div>
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
df = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'family_size', 'fare', 'embarked', 'survived']]
df.head()
```
<div class="output_subarea output_html rendered_html"><div>
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



## Save data
Finally, let's save our preprocessed data into a new CSV file to use later.
```python linenums="1"
# Saving dataframe to CSV
df.to_csv('processed_titanic.csv', index=False)
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


<!-- Citation -->
{% include "cite.md" %}