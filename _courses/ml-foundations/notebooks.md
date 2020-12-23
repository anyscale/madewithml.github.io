---
layout: page
title: Notebooks · ML Foundations
description: Learn how to use interactive notebooks for developing in Python.
image: /static/images/ml_foundations.png

course-url: /courses/ml-foundations/
next-lesson-url: /courses/ml-foundations/python/
---

<!-- Header -->
<div class="row">
  <div class="col-md-8 col-6 mr-auto">
    <h1 class="page-title">{{ page.title | split: " · " | first }}</h1>
  </div>
  <div class="col-md-4 col-6">
    <div class="btn-group float-right mb-0" role="group">
      <a href="{{ page.course-url }}" class="btn btn-sm btn-outline-secondary"><i
          class="fas fa-sm fa-arrow-left mr-1"></i>Return to course</a>
    </div>
  </div>
</div>
<hr class="mt-0">

In this lesson, we'll learn how to use interactive notebooks for developing in Python.

- [Set up](#setup)
- [Types of cells](#types)
- [Text cells](#text)
- [Running a cell](#run)
- [Editing a cell](#edit)
- [Moving a cell](#move)
- [Deleting a cell](#delete)
- [Code cells](#code)

> 📓 Follow along this lesson with the accompanying [notebook](https://colab.research.google.com/github/GokuMohandas/madewithml/blob/main/notebooks/01_Notebooks.ipynb){:target="_blank"}.

<h3 id="setup">Set up</h3>
1. Click on this link to open the accompanying [notebook](){:target="_blank"} for this lesson or create a blank one on [Google Colab](https://colab.research.google.com/){:target="_blank"}.
2. Sign into your [Google account](https://accounts.google.com/signin) to start using the notebook. If you don't want to save your work, you can skip the steps below. If you do not have access to Google, you can follow along using [Jupyter Lab](https://jupyter.org/).
3. If you do want to save your work, click the **COPY TO DRIVE** button on the toolbar. This will open a new notebook in a new tab. Rename this new notebook by removing the words Copy of from the title (change `Copy of 01_Notebooks` to "`1_Notebooks`).
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/notebooks/copy_to_drive.png" width="350">
    &emsp;&emsp;<img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/notebooks/rename.png" width="300">
</div>

<h3 id="types">Types of cells</h3>
Notebooks are made up of cells. There are two types of cells:

- `code cell`: used for writing and executing code.
- `text cell`: used for writing text, HTML, Markdown, etc.

<h3 id="text">Text cells</h3>
Click on a desired location in the notebook and create the cell by clicking on the `➕ TEXT` (located in the top left corner).

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/notebooks/text_cell.png" width="350">
</div>

Once you create the cell, click on it and type the following text inside it:

<pre class="output">
### This is a header
Hello world!
</pre>

<h3 id="run">Running a cell</h3>
Once you type inside the cell, press the `SHIFT` and `RETURN` (enter key) together to run the cell.

<h3 id="edit">Editing a cell</h3>
To edit a cell, double click on it and make any changes.

<h3 id="move">Moving a cell</h3>
Move a cell up and down by clicking on the cell and then pressing the ⬆ and ⬇ button on the top right of the cell.
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/notebooks/move_cell.png" width="550">
</div>

<h3 id="delete">Delete a cell</h3>
Delete the cell by clicking on it and pressing the trash can button 🗑️ on the top right corner of the cell. Alternatively, you can also press ⌘/Ctrl + M + D.
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/notebooks/delete_cell.png" width="550">
</div>

<h3 id="code">Code cells</h3>
Repeat the steps above to create and edit a code cell. You can create a code cell by clicking on the `➕ CODE` (located in the top left corner).
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/ml-foundations/notebooks/code_cell.png" width="350">
</div>
Once you've created the code cell, double click on it, type the following inside it and then press Shift + Enter to execute the code.
```python
print ("Hello world!")
```
<pre class="output">
Hello world!
</pre>

These are the basic concepts we'll need to use these notebooks but we'll learn few more tricks in subsequent lessons.

<!-- Footer -->
<hr>
<div class="row mb-4">
  <div class="col-6 mr-auto">
    <a href="{{ page.course-url }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-left mr-1"></i>Return to course</a>
  </div>
  <div class="col-6">
    <div class="float-right">
      <a href="{{ page.next-lesson-url }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-sm fa-arrow-right mr-1"></i>Next lesson</a>
    </div>
  </div>
</div>