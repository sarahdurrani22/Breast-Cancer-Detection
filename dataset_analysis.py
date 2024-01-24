import pandas as pd
from sklearn.datasets import load_breast_cancer
import ucimlrepo
from ucimlrepo import fetch_ucirepo

def fetch_and_describe_dataset():
    # Load the Breast Cancer Wisconsin Diagnostic dataset from scikit-learn
    breast_cancer = load_breast_cancer()

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)

    # Describe the dataset
    description = df.describe()

    # Print summary statistics of the dataset
    print(description)

def fetch_dataset_metadata():
    # Fetch dataset with ID 17 (Breast Cancer Wisconsin Diagnostic)
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    return breast_cancer_wisconsin_diagnostic