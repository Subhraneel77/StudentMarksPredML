## End to End Project

### Step 1: Create a new environment

```
conda create -p venv python==3.8

conda activate venv/
```
### Step 2: Create a .gitignore file

```
create the file by right click and include the venv in it
```

### Step 3: Create a requirements.txt file 
```
pip install -r requirements.txt
```

### Step 4: Create a setup.py file 
```
This is to install the entire project as a package. Additionally, write a function to read the packages from requirements.txt
```

### Step5: Create a folder `src` 
```
Include exception, logger, and utils python files. Make this folder as a package by including __init__.py file. The scr folder will include another folder with name components will be created. Include __init__.py also 
```
#### Step 5.1 Create a folder `components`

```
Include data_ingestion, data_transformation, model trainer, and __init_.py. These components are to be interconnected in future. 
```
#### Step 5.2 Create a folder called `pipeline`
```
Create two python files training_pipeline and prediction_pipeline with __init__.py folder
``` 

### Step 6: Create a folder called `notebooks` 
```
Create a folder called data and include the dataset. Additionally, create a EDA.ipynb file to do the EDA analysis.
```
# Student Marks Prediction Machine Learning Pipeline

## Overview

This project aims to predict student marks using machine learning techniques. The goal is to develop a robust pipeline that takes in various features related to students and predicts their marks accurately.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Dataset](#dataset)
4. [Preprocessing](#preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model Training](#model-training)
7. [Evaluation](#evaluation)
8. [Deployment](#deployment)
9. [Contributing](#contributing)
10. [License](#license)

## Installation

To run this project, you need to have Python 3.11 installed. Clone the repository and install the required dependencies using pip:

```bash
git clone https://github.com/yourusername/student-marks-prediction.git
cd student-marks-prediction
pip install -r requirements.txt
