---
template: lesson.html
title: Python Basics for Machine Learning
description: The basics of Python programming for machine learning.
keywords: python, decorators, functions, classes, mlops, applied ml, machine learning, ml in production, machine learning in production, applied machine learning
image: https://madewithml.com/static/images/ml_foundations.png
repository: https://github.com/GokuMohandas/madewithml
notebook: https://colab.research.google.com/github/GokuMohandas/madewithml/blob/main/notebooks/02_Python.ipynb
---

{% include "styles/lesson.md" %}

## Variables
Variables are containers for holding data and they're defined by a name and value.

<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/basics/python/variables.png" width="250">
</div>

```python linenums="1"
# Integer variable
x = 5
print (x)
print (type(x))
```
<pre class="output">
5
<class 'int'>
</pre>

!!! tip
    Here we use the variable name `x` in our examples but when you're working on a specific task, be sure to be explicit (ex. `first_name`) when creating variables (applies to functions, classes, etc. as well).

We can change the value of a variable by simply assigning a new value to it.

```python linenums="1"
# String variable
x = "hello"
print (x)
print (type(x))
```
<pre class="output">
hello
<class 'str'>
</pre>
There are many different types of variables: integers, floats, strings, boolean etc.
```python linenums="1"
# int variable
x = 5
print (x, type(x))
```
<pre class="output">
5 <class 'int'>
</pre>
```python linenums="1"
# float variable
x = 5.0
print (x, type(x))
```
<pre class="output">
5.0 <class 'float'>
</pre>
```python linenums="1"
# text variable
x = "5"
print (x, type(x))
```
<pre class="output">
5 <class 'str'>
</pre>
```python linenums="1"
# boolean variable
x = True
print (x, type(x))
```
<pre class="output">
True <class 'bool'>
</pre>
We can also do operations with variables:
```python linenums="1"
# Variables can be used with each other
a = 1
b = 2
c = a + b
print (c)
```
<pre class="output">
3
</pre>
We should always know what types of variables we're dealing with so we can do the right operations with them. Here's a common mistake that can happen if we're using the wrong variable type.
```python linenums="1"
# int variables
a = 5
b = 3
print (a + b)
```
<pre class="output">
8
</pre>
```python linenums="1"
# string variables
a = "5"
b = "3"
print (a + b)
```
<pre class="output">
53
</pre>

## Lists
Lists are an ordered, mutable (changeable) collection of values that are comma separated and enclosed by square brackets. A list can be comprised of many different types of variables. Below is a list with an integer, string and a float:

```python linenums="1"
# Creating a list
x = [3, "hello", 1.2]
print (x)
```
<pre class="output">
[3, 'hello', 1.2]
</pre>
```python linenums="1"
# Length of a list
len(x)
```
<pre class="output">
3
</pre>
We can add to a list by using the append function:
```python linenums="1"
# Adding to a list
x.append(7)
print (x)
print (len(x))
```
<pre class="output">
[3, 'hello', 1.2, 7]
4
</pre>
and just as easily replace existing items:
```python linenums="1"
# Replacing items in a list
x[1] = "bye"
print (x)
```
<pre class="output">
[3, 'bye', 1.2, 7]
</pre>
and perform operations with lists:
```python linenums="1"
# Operations
y = [2.4, "world"]
z = x + y
print (z)
```
<pre class="output">
[3, 'bye', 1.2, 7, 2.4, 'world']
</pre>

## Indexing
Indexing and slicing from lists allow us to retrieve specific values within lists. Note that indices can be positive (starting from 0) or negative (-1 and lower, where -1 is the last item in the list).
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/basics/python/indexing.png" width="300">
</div>

```python linenums="1"
# Indexing
x = [3, "hello", 1.2]
print ("x[0]: ", x[0])
print ("x[1]: ", x[1])
print ("x[-1]: ", x[-1]) # the last item
print ("x[-2]: ", x[-2]) # the second to last item
```
<pre class="output">
x[0]:  3
x[1]:  hello
x[-1]:  1.2
x[-2]:  hello
</pre>

```python linenums="1"
# Slicing
print ("x[:]: ", x[:]) # all indices
print ("x[1:]: ", x[1:]) # index 1 to the end of the list
print ("x[1:2]: ", x[1:2]) # index 1 to index 2 (not including index 2)
print ("x[:-1]: ", x[:-1]) # index 0 to last index (not including last index)
```
<pre class="output">
x[:]:  [3, 'hello', 1.2]
x[1:]:  ['hello', 1.2]
x[1:2]:  ['hello']
x[:-1]:  [3, 'hello']
</pre>

## Tuples
Tuples are collections that are ordered and immutable (unchangeable). You will use these to store values that will never be changed.
```python linenums="1"
# Creating a tuple
x = (3.0, "hello") # tuples start and end with ()
print (x)
```
<pre class="output">
(3.0, 'hello')
</pre>

```python linenums="1"
# Adding values to a tuple
x = x + (5.6, 4)
print (x)
```
<pre class="output">
(3.0, 'hello', 5.6, 4)
</pre>

```python linenums="1"
# Try to change (it won't work and you'll get an error)
x[0] = 1.2
```
<pre class="output">
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
----> 1 x[0] = 1.2
TypeError: 'tuple' object does not support item assignment
</pre>

## Dictionaries
Dictionaries are an unordered, mutable and indexed collection of key-value pairs. You can retrieve values based on the key and a dictionary cannot have two of the same keys.
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/basics/python/dictionaries.png" width="350">
</div>

```python linenums="1"
# Creating a dictionary
person = {'name': 'Goku',
          'eye_color': 'brown'}
print (person)
print (person['name'])
print (person['eye_color'])
```
<pre class="output">
{'name': 'Goku', 'eye_color': 'brown'}
Goku
brown
</pre>

```python linenums="1"
# Changing the value for a key
person['eye_color'] = 'green'
print (person)
```
<pre class="output">
{'name': 'Goku', 'eye_color': 'green'}
</pre>

```python linenums="1"
# Adding new key-value pairs
person['age'] = 24
print (person)
```
<pre class="output">
{'name': 'Goku', 'eye_color': 'green', 'age': 24}
</pre>

```python linenums="1"
# Length of a dictionary
print (len(person))
```
<pre class="output">
3
</pre>

## If statements
We can use `if` statements to conditionally do something. The conditions are defined by the words `if`, `elif` (which stands for else if) and `else`. We can have as many `elif` statements as we want. The indented code below each condition is the code that will execute if the condition is `True`.

```python linenums="1"
# If statement
x = 4
if x < 1:
    score = 'low'
elif x <= 4: # elif = else if
    score = 'medium'
else:
    score = 'high'
print (score)
```
<pre class="output">
medium
</pre>

```python linenums="1"
# If statement with a boolean
x = True
if x:
    print ("it worked")
```
<pre class="output">
it worked
</pre>

## Loops
- [For loops](#for)
- [While loops](#while)

### For loops
A `for` loop can iterate over a collection of values (lists, tuples, dictionaries, etc.) The indented code is executed for each item in the collection of values.
```python linenums="1"
# For loop
veggies = ["carrots", "broccoli", "beans"]
for veggie in veggies:
    print (veggie)
```
<pre class="output">
carrots
broccoli
beans
</pre>

When the loop encounters the break command, the loop will terminate immediately. If there were more items in the list, they will not be processed.
```python linenums="1"
# `break` from a for loop
veggies = ["carrots", "broccoli", "beans"]
for veggie in veggies:
    if veggie == "broccoli":
        break
    print (veggie)
```
<pre class="output">
carrots
</pre>

When the loop encounters the `continue` command, the loop will skip all other operations for that item in the list only. If there were more items in the list, the loop will continue normally.
```python linenums="1"
# `continue` to the next iteration
veggies = ["carrots", "broccoli", "beans"]
for veggie in veggies:
    if veggie == "broccoli":
        continue
    print (veggie)
```
<pre class="output">
carrots
beans
</pre>

### While loops
A `while` loop can perform repeatedly as long as a condition is `True`. We can use `continue` and `break` commands in `while` loops as well.
```python linenums="1"
# While loop
x = 3
while x > 0:
    x -= 1 # same as x = x - 1
    print (x)
```
<pre class="output">
2
1
0
</pre>

## Functions
Functions are a way to modularize reusable pieces of code. They're defined by the keyword `def` which stands for definition and they can have the following components.
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/basics/python/functions.png" width="350">
</div>

```python linenums="1"
# Define the function
def add_two(x):
    """Increase x by 2."""
    x += 2
    return x
```

Here are the components that may be required when we want to use the function. we need to ensure that the function name and the input parameters match with how we defined the function above.
<div class="ai-center-all">
    <img src="https://raw.githubusercontent.com/GokuMohandas/madewithml/main/images/basics/python/calling_functions.png" width="400">
</div>

```python linenums="1"
# Use the function
score = 0
new_score = add_two(x=score)
print (new_score)
```
<pre class="output">
2
</pre>

A function can have as many input parameters and outputs as we want.
```python linenums="1"
# Function with multiple inputs
def join_name(first_name, last_name):
    """Combine first name and last name."""
    joined_name = first_name + " " + last_name
    return joined_name
```

```python linenums="1"
# Use the function
first_name = "Goku"
last_name = "Mohandas"
joined_name = join_name(
    first_name=first_name, last_name=last_name)
print (joined_name)
```
<pre class="output">
Goku Mohandas
</pre>

!!! note
    It's good practice to always use keyword argument when using a function so that it's very clear what input variable belongs to what function input parameter. On a related note, you will often see the terms `*args` and `**kwargs` which stand for arguments and keyword arguments. You can extract them when they are passed into a function. The significance of the `*` is that any number of arguments and keyword arguments can be passed into the function.

```python linenums="1"
def f(*args, **kwargs):
    x = args[0]
    y = kwargs.get('y')
    print (f"x: {x}, y: {y}")
```
```python linenums="1"
f(5, y=2)
```
<pre class="output">
x: 5, y: 2
</pre>


## Classes

Classes are object constructors and are a fundamental component of object oriented programming in Python. They are composed of a set of functions that define the class and it's operations.

### Magic functions
Classes can be customized with magic functions like `__init__` and `__str__`, to enable powerful operations.

The `__init__` function is used when an instance of the class is initialized.
```python linenums="1"
# Creating the class
class Pet(object):
    """Class object for a pet."""

    def __init__(self, species, name):
        """Initialize a Pet."""
        self.species = species
        self.name = name
```
```python linenums="1"
# Creating an instance of a class
my_dog = Pet(species="dog",
             name="Scooby")
print (my_dog)
print (my_dog.name)
```
<pre class="output">
<__main__.Pet object at 0x7fe487e9c358>
Scooby
</pre>

The `print (my_dog)` command printed something not so relevant to us. Let's fix that with the `__str__` function.
```python linenums="1"
# Creating the class
# Creating the class
class Pet(object):
    """Class object for a pet."""

    def __init__(self, species, name):
        """Initialize a Pet."""
        self.species = species
        self.name = name

    def __str__(self):
        """Output when printing an instance of a Pet."""
        return f"{self.species} named {self.name}"
```
```python linenums="1"
# Creating an instance of a class
my_dog = Pet(species="dog",
             name="Scooby")
print (my_dog)
print (my_dog.name)
```
<pre class="output">
dog named Scooby
Scooby
</pre>

!!! note
    We'll be exploring additional built-in functions in subsequent notebooks (like `__len__`, `__iter__` and `__getitem__`, etc.) but if you're curious, here is a [tutorial](https://rszalski.github.io/magicmethods/){:target="_blank"} on more magic methods.


### Object functions
Besides these magic functions, classes can also have *object* functions.
```python linenums="1"
# Creating the class
class Pet(object):
    """Class object for a pet."""

    def __init__(self, species, name):
        """Initialize a Pet."""
        self.species = species
        self.name = name

    def __str__(self):
        """Output when printing an instance of a Pet."""
        return f"{self.species} named {self.name}"

    def change_name(self, new_name):
        """Change the name of your Pet."""
        self.name = new_name
```
```python linenums="1"
# Creating an instance of a class
my_dog = Pet(species="dog", name="Scooby")
print (my_dog)
print (my_dog.name)
```
<pre class="output">
dog named Scooby
Scooby
</pre>

```python linenums="1"
# Using a class's function
my_dog.change_name(new_name="Scrappy")
print (my_dog)
print (my_dog.name)
```
<pre class="output">
dog named Scrappy
Scrappy
</pre>


### Inheritance
We can also build classes on top of one another using inheritance, which allows us to inherit all the properties and methods from another class (the parent).
```python linenums="1"
class Dog(Pet):
    def __init__(self, species, name, breed):
        super().__init__("dog", name)
        self.breed = breed

    def __str__(self):
        return f"{self.breed} named {self.name}"
```
```python linenums="1"
scooby = Dog(species="dog", breed="Great Dane", name="Scooby")
print (scooby)
```
<pre class="output">
Great Dane named Scooby
</pre>
```python linenums="1"
scooby.change_name('Scooby Doo')
print (scooby)
```
<pre class="output">
Great Dane named Scooby Doo
</pre>

Notice how we inherited the initialized variables from the parent `Pet` class like species and name. We also inherited the `change_name` function. But for the `__str__` function, we define our own version to overwrite the `Pet` `__str__` function. We can similarly overwrite any object functions as well.


### Methods
There are two important decorator methods to know about when it comes to classes: `@classmethod` and `@staticmethod`. We'll learn about decorators in the next section below but these specific methods pertain to classes so we'll cover them here.

```python linenums="1"
class Dog(Pet):
    def __init__(self, name, breed):
        super().__init__(species="dog", name=name)
        self.breed = breed

    def __str__(self):
        return f"{self.breed} named {self.name}"

    @classmethod
    def from_dict(cls, d):
        return cls(name=d["name"], breed=d["breed"])

    @staticmethod
    def is_cute(breed):
        return True  # all animaals are cute!
```

A `@classmethod` allows us to create class instances by passing in the uninstantiated class itself (`cls`). This is a great way to create (or load) classes from objects (ie. dictionaries).

```python linenums="1"
# Create instance
d = {"name": "Cassie", "breed": "Border Collie"}
cassie = Dog.from_dict(d=d)
print(cassie)
```
<pre class="output">
Border Collie named Cassie
</pre>

A `@staticmethod` can be called from an uninstantiated class object so we can do things like this:
```python linenums="1"
# Static method
Dog.is_cute(breed="Border Collie")
```
<pre class="output">
True
</pre>

## Decorators
Recall that functions allow us to modularize code and reuse them. However, we'll often want to add some functionality before or after the main function executes and we may want to do this for many different functions. Instead of adding more code to the original function, we can use decorators!

- **decorators**: augment a function with pre/post-processing. Decorators wrap around the main function and allow us to operate on the inputs and or outputs.

Suppose we have a function called operations which increments the input value x by 1.
```python linenums="1"
def operations(x):
    """Basic operations."""
    x += 1
    return x
```
```python linenums="1"
operations(x=1)
```
<pre class="output">
2
</pre>

Now let's say we want to increment our input x by 1 before and after the operations function executes and, to illustrate this example, let's say the increments have to be separate steps. Here's how we would do it by changing the original code:
```python linenums="1"
def operations(x):
    """Basic operations."""
    x += 1
    x += 1
    x += 1
    return x
```
```python linenums="1"
operations(x=1)
```
<pre class="output">
4
</pre>

We were able to achieve what we want but we now increased the size of our `operations` function and if we want to do the same incrementing for any other function, we have to add the same code to all of those as well ... not very efficient. To solve this, let's create a decorator called `add` which increments `x` by 1 before and after the main function `f` executes.

### Creating a decorator
The decorator function accepts a function `f` which is the function we wish to wrap around, in our case, it's `operations()`. The output of the decorator is its `wrapper` function which receives the arguments and keyword arguments passed to function `f`.

Inside the `wrapper` function, we can:
1. extract the input parameters passed to function `f`.
2. make any changes we want to the function inputs.
3. function `f` is executed
4. make any changes to the function outputs
5. `wrapper` function returns some value(s), which is what the decorator returns as well since it returns `wrapper`.

```python linenums="1"
# Decorator
def add(f):
    def wrapper(*args, **kwargs):
        """Wrapper function for @add."""
        x = kwargs.pop('x') # .get() if not altering x
        x += 1 # executes before function f
        x = f(*args, **kwargs, x=x)
        x += 1 # executes after function f
        return x
    return wrapper
```
<pre class="output"></pre>
We can use this decorator by simply adding it to the top of our main function preceded by the `@` symbol.
```python linenums="1"
@add
def operations(x):
    """Basic operations."""
    x += 1
    return x
```
```python linenums="1"
operations(x=1)
```
<pre class="output">
4
</pre>

Suppose we wanted to debug and see what function actually executed with `operations()`.
```python linenums="1"
operations.__name__, operations.__doc__
```
<pre class="output">
('wrapper', 'Wrapper function for @add.')
</pre>
The function name and docstring are not what we're looking for but it appears this way because the `wrapper` function is what was executed. In order to fix this, Python offers `functools.wraps` which carries the main function's metadata.
```python linenums="1"
from functools import wraps
```
```python linenums="1"
# Decorator
def add(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        """Wrapper function for @add."""
        x = kwargs.pop('x')
        x += 1
        x = f(*args, **kwargs, x=x)
        x += 1
        return x
    return wrap
```
```python linenums="1"
@add
def operations(x):
    """Basic operations."""
    x += 1
    return x
```
```python linenums="1"
operations.__name__, operations.__doc__
```
<pre class="output">
('operations', 'Basic operations.')
</pre>
Awesome! We were able to decorate our main function `operation()` to achieve the customization we wanted without actually altering the function. We can reuse our decorator for other functions that may need the same customization!

!!! note
    This was a dummy example to show how decorators work but we'll be using them heavily during our [applied ML](https://madewithml.com/courses/mlops/) lessons. A simple scenario would be using decorators to create uniform JSON responses from each API endpoint without including the bulky code in each endpoint.


### Callbacks
Decorators allow for customized operations before and after the main function's execution but what about in between? Suppose we want to conditionally/situationally do some operations. Instead of writing a whole bunch of if-statements and make our functions bulky, we can use callbacks!

- **callbacks**: conditional/situational processing within the function.

Our callbacks will be classes that have functions with key names that will execute at various periods during the main function's execution. The function names are up to us but we need to invoke the same callback functions within our main function.
```python linenums="1"
# Callback
class x_tracker(object):
    def __init__(self, x):
        self.history = []
    def at_start(self, x):
        self.history.append(x)
    def at_end(self, x):
        self.history.append(x)
```
We can pass in as many callbacks as we want and because they have appropriately named functions, they will be invoked at the appropriate times.
```python linenums="1"
def operations(x, callbacks=[]):
    """Basic operations."""
    for callback in callbacks:
        callback.at_start(x)
    x += 1
    for callback in callbacks:
        callback.at_end(x)
    return x
```
```python linenums="1"
x = 1
tracker = x_tracker(x=x)
operations(x=x, callbacks=[tracker])
```
<pre class="output">
2
</pre>
```python linenums="1"
tracker.history
```
<pre class="output">
[1, 2]
</pre>

### Putting it all together
decorators + callbacks = powerful customization before, during and after the main function’s execution without increasing its complexity. We will be using this duo to create powerful ML training scripts that are highly customizable in future lessons.

```python linenums="1"
from functools import wraps
```
```python linenums="1"
# Decorator
def add(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        """Wrapper function for @add."""
        x = kwargs.pop('x') # .get() if not altering x
        x += 1 # executes before function f
        x = f(*args, **kwargs, x=x)
        # can do things post function f as well
        return x
    return wrap
```
```python linenums="1"
# Callback
class x_tracker(object):
    def __init__(self, x):
        self.history = [x]
    def at_start(self, x):
        self.history.append(x)
    def at_end(self, x):
        self.history.append(x)
```
```python linenums="1"
# Main function
@add
def operations(x, callbacks=[]):
    """Basic operations."""
    for callback in callbacks:
        callback.at_start(x)
    x += 1
    for callback in callbacks:
        callback.at_end(x)
    return x
```
```python linenums="1"
x = 1
tracker = x_tracker(x=x)
operations(x=x, callbacks=[tracker])
```
<pre class="output">
3
</pre>
```python linenums="1"
tracker.history
```
<pre class="output">
[1, 2, 3]
</pre>


<!-- Citation -->
{% include "cite.md" %}