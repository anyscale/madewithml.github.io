---
description: Learn how to use interactive notebooks for developing in Python.
image: "/static/images/ml_foundations.png"
---

:octicons-mark-github-16: [Repository](https://github.com/GokuMohandas/madewithml){:target="_blank"} · :octicons-book-24: [Notebook](https://colab.research.google.com/github/GokuMohandas/madewithml/blob/main/notebooks/01_Notebooks.ipynb){:target="_blank"}

In this lesson, we'll learn how to use interactive notebooks for developing in Python.

## Set up
1. Click on this link to open the accompanying [notebook](https://colab.research.google.com/github/GokuMohandas/madewithml/blob/main/notebooks/01_Notebooks.ipynb){:target="_blank"} for this lesson or create a blank one on [Google Colab](https://colab.research.google.com/){:target="_blank"}.
2. Sign into your [Google account](https://accounts.google.com/signin) to start using the notebook. If you don't want to save your work, you can skip the steps below. If you do not have access to Google, you can follow along using [Jupyter Lab](https://jupyter.org/).
3. If you do want to save your work, click the **COPY TO DRIVE** button on the toolbar. This will open a new notebook in a new tab. Rename this new notebook by removing the words Copy of from the title (change `Copy of 01_Notebooks` to `1_Notebooks`).
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/notebooks/copy_to_drive.png" width="350">
    &emsp;&emsp;<img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/notebooks/rename.png" width="300">
</div>

## Types of cells
Notebooks are made up of cells. There are two types of cells:

- `code cell`: used for writing and executing code.
- `text cell`: used for writing text, HTML, Markdown, etc.

## Text cells
Click on a desired location in the notebook and create the cell by clicking on the `➕ TEXT` (located in the top left corner).

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/notebooks/text_cell.png" width="350">
</div>

Once you create the cell, click on it and type the following text inside it:

```bash
### This is a header
Hello world!
```

## Run a cell
Once you type inside the cell, press the `SHIFT` and `RETURN` (enter key) together to run the cell.

## Edit a cell
To edit a cell, double click on it and make any changes.

## Move a cell
Move a cell up and down by clicking on the cell and then pressing the ⬆ and ⬇ button on the top right of the cell.
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/notebooks/move_cell.png" width="550">
</div>

## Delete a cell
Delete the cell by clicking on it and pressing the trash can button 🗑️ on the top right corner of the cell. Alternatively, you can also press ⌘/Ctrl + M + D.
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/notebooks/delete_cell.png" width="550">
</div>

## Code cells
Repeat the steps above to create and edit a code cell. You can create a code cell by clicking on the `➕ CODE` (located in the top left corner).
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/notebooks/code_cell.png" width="350">
</div>
Once you've created the code cell, double click on it, type the following inside it and then press Shift + Enter to execute the code.
```python linenums="1"
print ("Hello world!")
```
<pre class="output">
Hello world!
</pre>

These are the basic concepts we'll need to use these notebooks but we'll learn few more tricks in subsequent lessons.
