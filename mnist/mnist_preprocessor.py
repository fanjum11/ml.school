
import os
import numpy as np
import pandas as pd
import tempfile

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from pickle import dump


# This is the location where the SageMaker Processing job
# will save the input dataset.
BASE_DIR = "/opt/ml/mnist_processing"
TRAIN_DATA_FILEPATH = Path(BASE_DIR) / "input" / "mnist_train.csv"
TEST_DATA_FILEPATH = Path(BASE_DIR) / "input" / "mnist_test.csv"


def save_splits(base_dir, train, validation, test):
    """
    One of the goals of this script is to output the three
    dataset splits. This function will save each of these
    splits to disk.
    """
    
    train_path = Path(base_dir) / "train" 
    validation_path = Path(base_dir) / "validation" 
    test_path = Path(base_dir) / "test"
    
    print (f"train_path is {train_path}")
    
    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(train).to_csv(train_path / "train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(validation_path / "validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(test_path / "test.csv", header=False, index=False)
    

def save_pipeline(base_dir, pipeline):
    """
    Saves the Scikit-Learn pipeline that we used to
    preprocess the data.
    """
    pipeline_path = Path(base_dir) / "pipeline"
    pipeline_path.mkdir(parents=True, exist_ok=True)
    dump(pipeline, open(pipeline_path / "pipeline.pkl", 'wb'))
    

def generate_baseline_dataset(split_name, base_dir, X, y):
    """
    To monitor the data and the quality of our model we need to compare the 
    production quality and results against a baseline. To create those baselines, 
    we need to use a dataset to compute statistics and constraints. That dataset
    should contain information in the same format as expected by the production
    endpoint. This function will generate a baseline dataset and save it to 
    disk so we can later use it.
    
    """
    baseline_path = Path(base_dir) / f"{split_name}-baseline" 
    baseline_path.mkdir(parents=True, exist_ok=True)

    df = X.copy()
    
    # The baseline dataset needs a column containing the groundtruth.
    df["groundtruth"] = y
    df["groundtruth"] = df["groundtruth"].values.astype(str)
    
    # We will use the baseline dataset to generate baselines
    # for monitoring data and model quality. To simplify the process, 
    # we don't want to include any NaN rows.
    df = df.dropna()

    df.to_json(baseline_path / f"{split_name}-baseline.json", orient='records', lines=True)
    
    
def preprocess(base_dir, train_data_filepath, test_data_filepath):
    """
    Preprocesses the supplied raw dataset and splits it into a train, validation,
    and a test set.
    """
    
    df_train = pd.read_csv(train_data_filepath)
    df_test = pd.read_csv(test_data_filepath)
    
    
    X = df_train.copy()
    columns = list(X.columns)
    
    X = X.to_numpy()
    
    np.random.shuffle(X)
    train, validation, empty = np.split(X, [int(.7 * len(X)), int(1.0 * len(X))])
    
    X_train = pd.DataFrame(train, columns=columns)
    X_validation = pd.DataFrame(validation, columns=columns)
    X_test = df_test.copy() # assuming that test.csv has same columns as in train.csv; have to check for robustness 
    
    y_train = X_train.label
    y_validation = X_validation.label
    y_test = X_test.label
    
    label_encoder = LabelEncoder()
    
    y_train = label_encoder.fit_transform(y_train)
    y_validation = label_encoder.transform(y_validation)
    y_test = label_encoder.transform(y_test)
        
    X_train.drop(["label"], axis=1, inplace=True)
    X_validation.drop(["label"], axis=1, inplace=True)
    X_test.drop(["label"], axis=1, inplace=True)
    X_train = X_train/255.0
    X_validation = X_validation/255.0
    X_test = X_test/255.0 
    

    # Let's generate a dataset that we can later use to compute
    # baseline statistics and constraints about the data that we
    # used to train our model.
    
    ### THIS IS CRASHING THE VM; WHY IS THAT
    #generate_baseline_dataset("train", base_dir, X_train, y_train)
    print ("--------------------------------------------------11")
    
    # To generate baseline constraints about the quality of the
    # model's predictions, we will use the test set.
    ##generate_baseline_dataset("test", base_dir, X_test, y_test)
    
    
    # Transform the data using the Scikit-Learn pipeline.
   # X_train = preprocessor.fit_transform(X_train)
   # X_validation = preprocessor.transform(X_validation)
    # X_test = preprocessor.transform(X_test)
        
    train = np.concatenate((X_train, np.expand_dims(y_train, axis=1)), axis=1)
    validation = np.concatenate((X_validation, np.expand_dims(y_validation, axis=1)), axis=1)
    test = np.concatenate((X_test, np.expand_dims(y_test, axis=1)), axis=1)
   
    save_splits(base_dir, train, validation, test)
    #save_pipeline(base_dir, pipeline=preprocessor)
    print ("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    
    save_splits(base_dir, train, validation, test)
        

if __name__ == "__main__":
    preprocess(BASE_DIR, TRAIN_DATA_FILEPATH, TEST_DATA_FILEPATH)
