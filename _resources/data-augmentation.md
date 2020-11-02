---
layout: resource
title: Data augmentation
description: Curated resources for data augmentation.
image: /assets/images/resources/data-augmentation.png
tags: data-augmentation text-augmentation image-augmentation augmentation

image-credit: https://cdn-images-1.medium.com/max/1000/1*dJNlEc7yf93K4pjRJL55PA.png
topic-description: "Data augmentation is a set of techniques to increase both the quantity and diversity of the dataset without collecting more data.
The specific techniques are dependent on the application area (NLP, CV, audio, etc.)."

overviews:
    -
        title: A Survey on Image Data Augmentation for Deep Learning
        url: https://link.springer.com/article/10.1186/s40537-019-0197-0
    -
        title: A Visual Survey of Data Augmentation in NLP
        url: https://amitness.com/2020/05/data-augmentation-for-nlp/

tutorials:
    -
        title: Multi-target in Albumentations
        url: https://towardsdatascience.com/multi-target-in-albumentations-16a777e9006e?source=friends_link&sk=8c3579aa48cfea5e5c703bda6fe5451c
    -
        title: Data augmentation recipes in tf.keras image-based models
        url: https://sayak.dev/tf.keras/data_augmentation/image/2020/05/10/augmemtation-recipes.html

libraries:
    -
        title: Natural language processing
        libraries:
            -
                title: NLPAug
                description: data augmentation for NLP.
                repo: makcedward/nlpaug
            -
                title: TextAttack
                description: a framework for adversarial attacks, data augmentation, and model training in NLP.
                repo: QData/TextAttack
            -
                title: TextAugment
                description: text augmentation library.
                repo: dsfsi/textaugment
    -
        title: Computer vision
        libraries:
            -
                title: Imgaug
                description: image augmentation for machine learning experiments.
                repo: aleju/imgaug
            -
                title: Albumentations
                description: fast image augmentation library.
                repo: albumentations-team/albumentations
            -
                title: Augmentor
                description: image augmentation library in Python for machine learning.
                repo: mdbloice/Augmentor
            -
                title: Kornia.augmentation
                description: a module to perform data augmentation in the GPU.
                repo: kornia/kornia
            -
                title: SOLT
                description: data augmentation libarary for Deep Learning, which supports images, segmentation masks, labels and keypoints.
                repo: MIPT-Oulu/solt
    -
        title: Other
        libraries:
            -
                title: Snorkel
                description: system for generating training data with weak supervision.
                repo: snorkel-team/snorkel
            -
                title: DeltaPy⁠⁠
                description: tabular data augmentation and feature engineering.
                repo: firmai/deltapy
            -
                title: Audiomentations
                description: a Python library for audio data augmentation.
                repo: iver56/audiomentations
            -
                title: Tsaug
                description: a Python package for time series augmentation.
                repo: arundo/tsaug

miscellaneous:
    -
        title: "Automating Data Augmentation: Practice, Theory and New Direction"
        url: https://ai.stanford.edu/blog/data-augmentation/
    -
        title: Training Generative Adversarial Networks with Limited Data
        url: https://research.nvidia.com/publication/2020-06_Training-Generative-Adversarial
    -
        title: Test-Time Data Augmentation
        url: https://stepup.ai/test_time_data_augmentation/

---
