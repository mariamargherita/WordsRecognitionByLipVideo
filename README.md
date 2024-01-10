# WordsRecognitionByLipVideo

<br>

This GitHub repository showcases an academic project that focuses on classifying event data obtained from an event-based sensor - [Event Camera], also known as a neuromorphic sensor.

[Event Camera]: https://en.wikipedia.org/wiki/Event_camera

Please note that the dataset was leveraged for a [Kaggle](https://www.kaggle.com/competitions/smemi309-final-evaluation-challenge-2022) challenge but it is not available anymore due to ownership rights. However, please feel free to reach out in case you are interested in the dataset.

This project was carried out in PyCharm, thus it is optimized for it. However, this should not keep you from using your own preferred server.
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

> **Problem**: Our goal is to construct a classifier that can determine the class of a new, unseen example by enhancing classifiers available at the [Kaggle competition link](https://www.kaggle.com/competitions/smemi309-final-evaluation-challenge-2022).

The main *metric* that will be used to assess the performance of the models is *accuracy*.
<br>

## Usage

To use this project, follow these steps:

1. **Clone the repository**: First, clone this repository to your local machine using

    ```bash
    git clone https://github.com/mariamargherita/WordsRecognitionByLipVideo.git
    ```

2. **Obtain the dataset**: To obtain the dataset please reach out. Unfortunately, the data are not publicly available due to ownership rights. \
The train data will have the following structure:
    ```bash
    local_repo/
    ├──── train10/
    │       ├── train10/
    │             ├── Addition/
    │             ├── Carnaval/
    │             ├── Decider/
    │             ├── Ecole/
    │             ├── Fillette/
    │             ├── Huitre/
    │             ├── Joyeux/
    │             ├── Musique/
    │             ├── Pyjama/
    │             └── Ruisseau/
    ├──── .venv/
    ├──── .gitignore
    ├──── .LICENSE
    ├──── ...
    └──── *.ipynb

    ```
Every folder within `train10/train10/` holds 32 csv files, named from `0.csv` to `31.csv`. These files represent event data focused on the face of a speaker uttering a specific french word, which is also the name of the parent folder.

3. **Install virtual environment**: This project requires a certain Conda environment. You can install it by typing the following in your terminal:

    ```bash
    conda env create -f lip_video_env.yml
    ```
   
You should now be able to run the Notebooks.

<br>

## Project Outline

### Data Preprocessing

The initial phase of the project involves preprocessing the raw event data. In the preprocessing phase, we first implement noise reduction methods to refine the event data. 
> **Enhancements**
> - Not yet started, will try noise reduction.

### Model Selection and Training

After preprocessing and exploring the data, the next step is model selection and training. We will be comparing the performance of a 
Bagging Random Forest versus the one of a CNN-LSTM.

> **Enhancements on CNN-LSTM**
> - The model was trained with 90% of training set and 10% of validation data to increase training data and limit overfitting
> - Lowering early stopping patience on validation loss from 10 to 5 to limit overfitting

> **Enhancements on Bagging Random Forest**
> - Not yet started

<br>

## Results


## Contributions

