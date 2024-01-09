# WordsRecognitionByLipVideo

<br>

This GitHub repository showcases an academic project that focuses on classifying event data obtained from an event-based sensor - [Event Camera], also known as a neuromorphic sensor.

[Event Camera]: https://en.wikipedia.org/wiki/Event_camera

Please note that the dataset was leveraged for a [Kaggle](https://www.kaggle.com/competitions/smemi309-final-evaluation-challenge-2022/data) challenge but it is not available anymore due to ownership rights. However, please feel free to reach out in case you are interested in the dataset.

<br>

## Introduction

An event is a 4-tuple $(x,y,p,t)$ where

- $(x,y)$ denotes the pixel's position associated with the event.
- $p$ is a boolean indicating whether the change in luminosity is increasing or decreasing.
- $t$ represents the timestamp (in $\mu s$) from the start of the recording.

Event Data are DataFrames, with each row representing an event, sorted in ascending order *w.r.t.* the timestamp.

> **Note**: In our unique hardware configuration provided by the manufacturer, the range of $x$ is from $0$ to $480$, $y$ varies from $0$ to $640$, $p$ can be either $0$ (decrease of luminosity) or $1$ (increase of luminosity), and $t$ is a floating-point number.

<br>

## Project Objective

The primary goal of this project is to address the following problem:

> **Problem**: Given 10 distinct classes, each with 32 examples, our goal is to construct a classifier that can accurately determine the class of a new, unseen example.

The main *metric* that will be used to assess the performance of the models is *accuracy*.\
This problem is the central focus of our project and all subsequent work will be aimed at solving it.

<br>