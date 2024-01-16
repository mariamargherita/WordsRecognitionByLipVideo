# WordsRecognitionByLipVideo

<br>

This GitHub repository showcases a project that focuses on classifying event data obtained from an event-based sensor - [Event Camera], also known as a neuromorphic sensor.

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

> **Problem**: Our goal is to construct a classifier that can determine the class of a new, unseen example as accurately as possible. We will compare then its performance with the one of a CNN-LSTM.

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

The initial phase of the project involves preprocessing the raw event data. In the preprocessing steps we performed noise reduction which did not result in an improvement in the model performance.
This is probably due to the fact that in our use case noise makes our model more general. For this reason, we decided not to reduce noise in our data.

> **Note**: The code for noise reduction implementation was left in the repository for reference.
> **Note**: The preprocessing steps performed are different for the two models. This is due to the fact that some steps made in the preprocessing pipeline of the CNN-LSTM are valuable to improve its performance but are not needed for improving other classification models' accuracy (i.e. mini-batches creation).

<br>

### Model Selection and Training

After preprocessing and exploring the data, the next step is model selection and training. We will be comparing the performance of a Random Forest versus the one of a CNN-LSTM.

For the CNN-LSTM we tried different model complexities and tuned parameters and hyperparameters. We also tried different test sizes and batch sizes since
we do not have much data so this could have a strong impact on how well the neural network learns to generalize. Finally, we made sure to add dropout and 
early stopping to limit over fitting.

<br>

## Results

From the CNN-LSTM we get a $93% accuracy$ on the test set. To achieve this, we trained the model on 90% of training data and reserved a
10% for validation data. Once we found the model with the best performance on the validation data, we trained the best model on the full training data and
predicted the test data labels, getting a test accuracy of 93%.

<br>

## Contributions

Here are some of the steps that could still be taken in order to potentially improve the models:

- CNN-LSTM:
  - Add regularization to further limit over fitting
  - Add attention layers in neural network composition
  - Try to tune Adam optimizer's parameters (i.e. learning rate)
- Random Forest:
  - 
