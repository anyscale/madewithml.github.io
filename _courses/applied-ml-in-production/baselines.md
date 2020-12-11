---
layout: page
title: Baselines · Applied ML in Production
description: Motivating the use of baselines for iterative modeling.
image: /static/images/courses/applied-ml-in-production/baselines.png
tags: baselines modeling

course-url: /courses/applied-ml-in-production/
next-lesson-url: /courses/applied-ml-in-production/random/
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

<!-- Video -->
<!-- <div class="ai-center-all mt-2">
    <iframe width="600" height="337.5" src="https://www.youtube.com/embed/i524r-nZi6Q?rel=0" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen></iframe>
</div>
<div class="ai-center-all mt-2">
  <small>Accompanying video for this lesson. <a href="https://www.youtube.com/madewithml?sub_confirmation=1" target="_blank">Subscribe</a> for updates!</small>
</div>

<div class="alert info mt-4" role="alert">
  <span style="text-align: left;">
    <i class="fas fa-info-circle mr-1"></i> Connect with the author, <i>Goku Mohandas</i>, on
    <a href="https://twitter.com/GokuMohandas" target="_blank">Twitter</a> and
    <a href="https://www.linkedin.com/in/goku" target="_blank">LinkedIn</a> for
    interactive conversations on upcoming lessons.
  </span>
</div> -->

<h3><u>Intuition</u></h3>

**What are they?** Baselines are simple benchmarks which pave the way for iterative development.

**Why do we need them?**
- Rapid experimentation via hyperparameter tuning thanks to low model complexity.
- Discovery of data issues, false assumptions, bugs in code, etc. since model itself is not complex.
- [Pareto's principle](https://en.wikipedia.org/wiki/Pareto_principle){:target="_blank"}: we can achieve decent performance with minimal initial effort.

**How do we use them?**
1. Start with the simplest possible baseline to compare subsequent development with. This is often a random (chance) model.
2. Develop a rule-based approach (when possible) using IFTT, auxiliary data, etc.
3. Slowly add complexity by *addressing* limitations and *motivating* representations and model architectures.
4. Weigh *tradeoffs* (performance, latency, size, etc.) between performant baselines.
5. Revisit and iterate on baselines as your dataset grows.

> 🔄 You can also baseline on your dataset. Instead of using a fixed dataset and iterating on the models, choose a good baseline and iterate on the dataset.
- remove or fix data samples (FP, FN)
- prepare and transform features
- expand or consolidate classes
- auxiliary datasets

**Tradeoffs to consider?**
When choosing what model architecture(s) to proceed with, there are a few important aspects to consider:
- `performance`: consider overall and fine-grained (ex. per-class) performance.
- `latency`: how quickly does your model respond for inference.
- `size`: how large is your model and can you support it's storage.
- `compute`: how much will it cost ($, carbon footprint, etc.) to train your model?
- `interpretability`: does your model need to explain its predictions?
- `bias checks`: does your model pass key bias checks?
- `🕓 to develop`: how long do you have to develop the first version?
- `🕓 to retrain`: how long does it take to retrain your model? This is very important to consider if you need to retrain often.
- `maintenance overhead`: who and what will be required to maintain your model versions because the real work with ML begins after deploying v1. You can't just hand it off to your site reliability team to maintain it like many teams do with traditional software.

<!-- <i class="fab fa-youtube ai-color-youtube mr-1"></i> Watch from [0:00](https://www.youtube.com/watch?v=i524r-nZi6Q&list=PLqy_sIcckLC2jrxQhyqWDhL_9Uwxz8UFq&index=8&t=0s){:target="_blank"} for a video walkthrough of this section. -->

<h3><u>Application</u></h3>

> The notebook for this section can be found [here](https://github.com/madewithml/applied-ml-in-production/blob/master/notebooks/tagifai.ipynb){:target="_blank"}.

Each application's baseline trajectory varies based on the task and motivations. For our application, we're going to follow this path:
1. [Random](#random)
2. [Rule-based](#rule-based) (using stemmed aliases and parents)
3. [Simple ML](#simple-ml) (logistic regression, KNN, GBM, etc.)
4. [CNN w/ embeddings](#cnn)
5. [RNN w/ embeddings](#rnn)
6. [Transformers w/ contextual embeddings](#transformers)

We'll motivate the need for slowly adding complexity from both the *representation* (ex. embeddings) and *architecture* (ex. CNNs) views, as well as address the limitation at each step of the way.

> <i class="fab fa-youtube ai-color-youtube mr-1"></i> There's a lot of code to cover here, so instead of writing about it, it's much easier to walk through the code. So be sure to watch the accompanying video because we cover the code below in detail.

<hr>

<h4 id="random">Random</h4>

<u><i>motivation</i></u>: We want to know what random (chance) performance looks like. All of our efforts should be well above this.
```json
{
  "precision": 0.061257475827160714,
  "recall": 0.4951782007674865,
  "f1": 0.1002276753129853,
  "num_samples": 473.0
}
```
<u><i>limitations</i></u>: we didn't use the tokens in our input to affect our predictions so nothing was learned.

<hr>

<h4 id="rule-based">Rule-based</h4>

<u><i>motivation</i></u>: we want to use signals in our inputs (along with domain expertise and auxiliary data) to determine the labels.
```json
{
    "precision": 0.8529778980663546,
    "recall": 0.44442929438262596,
    "f1": 0.5521801514946141,
    "num_samples": 461.0
}
```
<u><i>limitations</i></u>: we failed to generalize or learn any implicit patterns to predict the labels because we treat the tokens in our input as isolated entities.

> We would ideally spend more time tuning our model because it's so simple and quick to train. This approach also applies to all the other models we'll look at as well.


<hr>

<h4 id="simple-ml">Simple ML</h4>
<u><i>motivation</i></u>:
- *representation*: use term frequency-inverse document frequency [(TF-IDF)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf){:target="_blank"} to capture the significance of a token to a particular input with respect to all the inputs, as opposed to treating the words in our input text as isolated tokens.
- *architecture*: we want our model to meaningfully extract the encoded signal to predict the output labels.

```json
{
  "logistic-regression": {
    "precision": 0.36304785843956355,
    "recall": 0.08232620273814854,
    "f1": 0.1264879417481289,
    "num_samples": 473.0
  },
  "k-nearest-neighbors": {
    "precision": 0.6820960409682214,
    "recall": 0.37595961284200197,
    "f1": 0.4623229605088978,
    "num_samples": 473.0
  },
  "random-forest": {
    "precision": 0.7518445839874411,
    "recall": 0.25277614134633836,
    "f1": 0.35108409502730287,
    "num_samples": 473.0
  },
  "gradient-boosting-machine": {
    "precision": 0.8171735969072024,
    "recall": 0.551769809487174,
    "f1": 0.6270937317892625,
    "num_samples": 473.0
  },
  "support-vector-machine": {
    "precision": 0.8838762228347793,
    "recall": 0.43030260434693934,
    "f1": 0.5406234782974617,
    "num_samples": 473.0
  }
}
```
<u><i>limitations</i></u>:
- *representation*: TF-IDF representations don't encapsulate much signal beyond frequency but we require more fine-grained token representations.
- *architecture*: we want to develop models that can use better represented encodings in a more contextual manner.

<hr>

<h4 id="cnn">CNN w/ Embeddings</h4>
<u><i>motivation</i></u>:
- *representation*: we want to have more robust (split tokens to characters) and meaningful ([embeddings](https://github.com/madewithml/basics/blob/master/notebooks/14_Embeddings/14_PT_Embeddings.ipynb){:target="_blank"}) representations for our input tokens.
- *architecture*: we want to process our encoded inputs using [convolution (CNN)](https://github.com/madewithml/basics/blob/master/notebooks/13_Convolutional_Neural_Networks/13_PT_Convolutional_Neural_Networks.ipynb){:target="_blank"} filters that can learn to analyze windows of embedded tokens to extract meaningful signal.

```json
{
  "precision": 0.890981133343249,
  "recall": 0.5151039412831285,
  "f1": 0.6240891757280376,
  "num_samples": 473.0
}
```

<u><i>limitations</i></u>:
- *representation*: embeddings are not contextual.
- *architecture*: extracting signal from encoded inputs is limited by filter widths.

> Since we're dealing with simple architectures and fast training times, it's a good opportunity to explore tuning and experiment with k-fold cross validation to properly reach any conclusions about performance.

<hr>

<h4 id="rnn">RNN w/ Embeddings</h4>
<u><i>motivation</i></u>: let's see if processing our embedded tokens in a sequential fashion using [recurrent neural networks (RNNs)](https://github.com/madewithml/basics/blob/master/notebooks/15_Recurrent_Neural_Networks/15_PT_Recurrent_Neural_Networks.ipynb){:target="_blank"} can yield better performance.

```json
{
  "precision": 0.3565450372023267,
  "recall": 0.23188423604310304,
  "f1": 0.2490844416412557,
  "num_samples": 473.0
}
```

<u><i>limitations</i></u>: since we're using character embeddings our encoded sequences are quite long (>100), the RNNs may potentially be suffering from memory issues. We also can't process our tokens in parallel because we're restricted by sequential processing.

> Don't be afraid to experiment with stacking models if they're able to extract unique signal from your encoded data, for example applying CNNs on the outputs from the RNN (outputs from all tokens, not just last relevant one).

<hr>

<h4 id="transformers">Transformers w/ Contextual Embeddings</h4>
<u><i>motivation</i></u>:
- *representation*: we want better representation for our input tokens via contextual embeddings where the token representation is based on the specific neighboring tokens. We can also use sub-word tokens, as opposed to character or word tokens, since they can hold more meaningful representations for many of our keywords, prefixes, suffixes, etc. without having to use filters with specific widths.
- *architecture*: we want to use [Transformers](https://www.youtube.com/watch?v=LwV7LKunDbs){:target="_blank"} to attend (in parallel) to all the tokens in our input, as opposed to being limited by filter spans (CNNs) or memory issues from sequential processing (RNNs).

> We will only be used the encoder from a [pretrained transformer](https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel){:target="_blank"} as a feature extractor.

```json
{
  "precision": 0.8516848693634407,
  "recall": 0.6078315390138052,
  "f1": 0.6813739640586111,
  "num_samples": 473.0
}
```

<u><i>limitations</i></u>: transformers can be quite large and we'll have to weigh tradeoffs before deciding on a model.

#### Tradeoffs
We're going to go with the embeddings via CNN approach and optimize it because performance is quite similar to the contextualized embeddings via transformers approach but at much lower cost.

> This was just one run on one split so you'll want to experiment with k-fold cross validation to properly reach any conclusions about performance. Also make sure you take the time to explore tuning for these baselines since their training periods are quite fast (we can achieve f1 of 0.7 with just a bit of tuning for both architectures). We should also benchmark on other important metrics as we iterate, not just precision and recall.

```yaml
CNN: 4.2 MB
Transformer: 439.9 MB
```

> Interpretability was not one of requirements but note that we could've tweaked model outputs to deliver it. For example, since we used SAME padding for our CNN, we can use the activation scores to extract influential n-grams. Similarly, we could have used self-attention weights from our Transformer encoder to find influential sub-tokens.

> <i class="fab fa-youtube ai-color-youtube mr-1"></i> Be sure to watch the accompanying video because we cover the baselines in detail.



<h3><u>Resources</u></h3>
- [Backing off towards simplicity - why baselines need more love](https://smerity.com/articles/2017/baselines_need_love.html){:target="_blank"}
- [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://arxiv.org/abs/1811.12808){:target="_blank"}


<!-- <i class="fab fa-youtube ai-color-youtube mr-1"></i> Watch from [21:10](https://www.youtube.com/watch?v=i524r-nZi6Q&list=PLqy_sIcckLC2jrxQhyqWDhL_9Uwxz8UFq&index=8&t=1270s){:target="_blank"} for a video walk-through of this section. -->


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