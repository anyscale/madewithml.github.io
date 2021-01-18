---
title: Examples
description: Examples of what you can do with mkdocs.
---


# Examples

## Text
The `#!python range()` function is used to generate a sequence of numbers. This is what `#!js backticks` will look like.

Press ++ctrl+alt+del++ to exit.

> This is a blockquote

## Icons
- [Octicons](https://primer.style/octicons/){:target="_blank"}
    - :octicons-mark-github-16: Code · :octicons-note-24: Notebook · :octicons-device-camera-video-24: Video
- [https://material.io/resources/icons/](){:target="_blank"}
    - :material-check:
- [https://fontawesome.com/icons](){:target="_blank"}
    - :fontawesome-brands-twitter:


## MathJAX

$$
\operatorname{ker} f=\{g\in G:f(g)=e_{H}\}{\mbox{.}}
$$

The homomorphism $f$ is injective if and only if its kernel is only the
singleton set $e_G$, because otherwise $\exists a,b\in G$ with $a\neq b$ such
that $f(a)=f(b)$.

## Images
<figure>
  <img src="https://dummyimage.com/600x400/eee/aaa" width="300" />
  <figcaption>Image caption</figcaption>
</figure>

## Lists

### Ordered lists
1. Vivamus id mi enim. Integer id turpis sapien. Ut condimentum lobortis
  sagittis. Aliquam purus tellus, faucibus eget urna at, iaculis venenatis
  nulla. Vivamus a pharetra leo.

    1. Vivamus venenatis porttitor tortor sit amet rutrum. Pellentesque aliquet
      quam enim, eu volutpat urna rutrum a. Nam vehicula nunc mauris, a
      ultricies libero efficitur sed.

    2. Morbi eget dapibus felis. Vivamus venenatis porttitor tortor sit amet
      rutrum. Pellentesque aliquet quam enim, eu volutpat urna rutrum a.

        1. Mauris dictum mi lacus
        2. Ut sit amet placerat ante
        3. Suspendisse ac eros arcu

### User defined lists
`Lorem ipsum dolor sit amet`
:   Sed sagittis eleifend rutrum. Donec vitae suscipit est. Nullam tempus
    tellus non sem sollicitudin, quis rutrum leo facilisis.

**Cras arcu libero**
:   Aliquam metus eros, pretium sed nulla venenatis, faucibus auctor ex. Proin
    ut eros sed sapien ullamcorper consequat. Nunc ligula ante.

    Duis mollis est eget nibh volutpat, fermentum aliquet dui mollis.
    Nam vulputate tincidunt fringilla.
    Nullam dignissim ultrices urna non auctor.

### Task lists
- [x] Lorem ipsum dolor sit amet, consectetur adipiscing elit
- [ ] Vestibulum convallis sit amet nisi a tincidunt
    * [x] In hac habitasse platea dictumst
    * [x] In scelerisque nibh non dolor mollis congue sed et metus
    * [ ] Praesent sed risus massa
- [ ] Aenean pretium efficitur erat, donec pharetra, ligula non scelerisque

## Code
```python linenums="1"
# Load projects
url = "https://raw.githubusercontent.com/madewithml/projects.json"
projects = json.loads(urlopen(url).read())
print (json.dumps(projects[-305], indent=2))
```
<pre class="output">
{
  "id": 324,
  "title": "AdverTorch",
  "description": "A Toolbox for Adversarial Robustness Research",
  "tags": [
    "code",
    "library",
    "security",
    "adversarial-learning",
    "adversarial-attacks",
    "adversarial-perturbations"
  ]
}
</pre>
``` python linenums="1" hl_lines="2 3"
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```

## Dataframes
<div class="output_subarea output_html rendered_html"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>leukocyte_count</th>
      <th>blood_pressure</th>
      <th>tumor_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15.335860</td>
      <td>14.637535</td>
      <td>benign</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.857535</td>
      <td>14.518942</td>
      <td>malignant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17.632579</td>
      <td>15.869585</td>
      <td>benign</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18.369174</td>
      <td>14.774547</td>
      <td>benign</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.509367</td>
      <td>15.892224</td>
      <td>malignant</td>
    </tr>
  </tbody>
</table>
</div></div>

## Tables
| Method      | Description                          |
| :---------- | :----------------------------------- |
| `GET`       | :material-check:     Fetch resource  |
| `PUT`       | :material-check-all: Update resource |
| `DELETE`    | :material-close:     Delete resource |

## Buttons
[:material-calendar: &nbsp; Subscibe](https://madewithml.com/subscribe/){: .md-button .md-button--primary }

## Footnotes
Lorem ipsum[^1] dolor sit amet, consectetur adipiscing elit.[^2]
[^1]: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
[^2]:
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

## Alerts
Alert with a title

!!! note "Phasellus posuere in sem ut cursus"
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

Alert without a title

!!! success ""
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

Alert with nested code

!!! danger
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

    ``` python
    def bubble_sort(items):
        for i in range(len(items)):
            for j in range(len(items) - 1 - i):
                if items[j] > items[j + 1]:
                    items[j], items[j + 1] = items[j + 1], items[j]
    ```

    Nunc eu odio eleifend, blandit leo a, volutpat sapien. Phasellus posuere in
    sem ut cursus. Nullam sit amet tincidunt ipsum, sit amet elementum turpis.
    Etiam ipsum quam, mattis in purus vitae, lacinia fermentum enim.