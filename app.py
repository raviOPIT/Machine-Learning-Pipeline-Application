import pandas as pd
import numpy as np
import argparse
import joblib
import warnings
from typing import Optional
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError

# Suppress all warnings
warnings.filterwarnings("ignore")

class MLApplication:
    """
    A class to manage the end-to-end machine learning workflow, including data loading, preprocessing, 
    model training, evaluation, and saving the trained pipeline.

    Purpose:
    --------
    This class simplifies the implementation of machine learning pipelines for both classification 
    and regression tasks by providing a modular framework.

    Features:
    ---------
    - Supports multiple algorithms for classification and regression.
    - Handles preprocessing for numeric and categorical features.
    - Automatically determines the task type (classification or regression) based on the target column.
    - Includes cross-validation for robust evaluation of models.
    - Enables easy saving and reuse of trained pipelines.

    Usage:
    ------
    Create an instance of the `MLApplication` class by providing the required parameters like the 
    path to the dataset, target column, and algorithm. Call the methods in sequence to execute the pipeline.
    Example:
        app = MLApplication(csv_path="data.csv", target_column="target", algorithm="random_forest")
        app.load_data()
        app.preprocess_data()
        app.build_pipeline()
        app.train_and_evaluate()
        app.save_pipeline(output_path="model_pipeline.pkl")
    """
    def __init__(self, 
                 csv_path: str, 
                 target_column: str, 
                 algorithm: str, 
                 random_state: Optional[int] = 42, 
                 num_folds: Optional[int] = 5):
        """
        Initialize the machine learning application.

        Parameters:
        - csv_path (str): Path to the CSV file containing the dataset. It must be a valid file path, 
          and the file should be in CSV format with a tabular structure.
        - target_column (str): Name of the target column in the dataset.
        - algorithm (str): Name of the algorithm to use (e.g., 'logistic_regression', 'random_forest').
        - random_state (Optional[int]): Random state for reproducibility. Must be a non-negative integer. Defaults to 42.
        - num_folds (Optional[int]): Number of folds for cross-validation. Must be an integer greater than 1. Defaults to 5.
        """
        self.csv_path: str = csv_path  # File path to the dataset
        self.target_column: str = target_column  # Target column name in the dataset
        self.algorithm: str = algorithm  # Algorithm name for the ML task
        self.random_state: Optional[int] = random_state  # Random state for reproducibility
        self.num_folds: Optional[int] = num_folds  # Number of folds for cross-validation

        self.model: Optional[Pipeline] = None  # Placeholder for the ML model
        self.pipeline: Optional[Pipeline] = None  # Placeholder for the ML pipeline
        self.X: Optional[pd.DataFrame] = None  # Features (input variables for training)
        self.y: Optional[pd.Series] = None  # Target variable (output variable for training)
        self.preprocessor: Optional[ColumnTransformer] = None  # Preprocessing pipeline (handles data transformations)
        self.task_type: Optional[str] = None  # Task type: 'classification' or 'regression'

    def load_data(self) -> None:
        """
        Load data from the CSV file and determine the task type based on the target column's data type.

        Expected Structure of CSV:
        - The dataset should be in a tabular format with the target column explicitly named.
        - Feature columns can include both numeric and categorical data types.
        - Missing values in feature columns are allowed but will be handled during preprocessing.
        - The target column should contain numeric values for regression tasks or categorical values for classification tasks.

        Raises:
        - ValueError: If the dataset is empty or the target column's data type is unsupported or if the target column does not exist.
        """
        print("Starting data loading...")
        try:
            # Attempt to load the dataset from the given CSV path
            data: pd.DataFrame = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            raise FileNotFoundError("The specified file path does not exist. Please check the file path and try again.")
        except pd.errors.EmptyDataError:
            raise ValueError("The dataset is empty or contains no columns.")
        except Exception as e:
            raise ValueError(f"An unexpected error occurred while loading the data: {e}")

        if data.empty:
            raise ValueError("The dataset is empty.")
        
        # Check if target column exists
        if self.target_column not in data.columns:
            raise KeyError(f"Column '{self.target_column}' does not exist in the dataset. Available columns are: {list(data.columns)}")
        
        # Split the dataset into features (X) and target (y)
        try:
            self.y = data[self.target_column]  # Extract the target column
            self.X = data.drop(columns=[self.target_column])  # Extract feature columns
        except Exception as e:
            raise ValueError(f"An unexpected error occurred while separating the features and target variable from the data: {e}")
        # Determine task type based on the target column's data type 
        # if isinstance(self.y.dtype, pd.CategoricalDtype) or self.y.dtype == 'object' or self.y.dtype =='bool'or len(self.y.unique()) < 10: #Assumes categorical if < 10 unique values as a thumb of rule
        if isinstance(self.y.dtype, pd.CategoricalDtype) or self.y.dtype == 'object' or self.y.dtype =='bool': #Assumes categorical if < 10 unique values as a thumb of rule
            self.task_type = 'classification'  # Categorical or object target indicates a classification task
        elif pd.api.types.is_numeric_dtype(self.y.dtype):
            self.task_type = 'regression' # Numeric target indicates a regression task
        else:
            raise ValueError("The target column data type is not supported for classification or regression.")
        print(f"Task type determined: {self.task_type}")

    def preprocess_data(self) -> None:
        """
        Preprocess the dataset by handling missing values and applying transformations to numeric and categorical features.

        Steps performed:
        ----------------
        1. **Handling Missing Values:**
           - Numeric columns: Missing values are imputed using the median. Edge case: If all values in a numeric column are NaN, imputation will result in a single default value (median).
           - Categorical columns: Missing values are imputed using the most frequent category. Edge case: If a categorical column has all unique values, imputation defaults to one of the modes.

        2. **Scaling Numeric Features:**
           - Numeric features are scaled to have zero mean and unit variance using StandardScaler. Note: Outliers can significantly affect scaling.

        3. **Encoding Categorical Features:**
           - Categorical features are one-hot encoded, creating binary columns for each category. Default behavior: Unknown categories in new data will be ignored due to `handle_unknown='ignore'.`

        These transformations are combined into a ColumnTransformer to ensure appropriate preprocessing for each feature type.

        Raises:
        - ValueError: If unsupported data types are found or if columns contain only NaN values.
        """
        print("Starting preprocessing...")
        # Identify columns that contain only NaN values
        all_nan_cols = self.X.columns[self.X.isna().all()]
        if not all_nan_cols.empty:
            raise ValueError(f"The following columns contain only NaN values and cannot be processed: {list(all_nan_cols)}")

        if self.y.isna().all():
            raise ValueError(f"The target column contains only NaN values and cannot be processed")

        
        # Check for unsupported data types in the dataset
        unsupported_cols = self.X.select_dtypes(exclude=["int64", "float64", "object", "bool", "category"])
        if not unsupported_cols.empty:
            raise ValueError(f"Unsupported data types found in features: {unsupported_cols.dtypes.to_dict()}")
        
        # Identify numeric and categorical columns in the dataset
        numeric_features = self.X.select_dtypes(include=['int64', 'float64']).columns  # Numeric columns
        categorical_features = self.X.select_dtypes(include=['object', 'bool', 'category']).columns  # Categorical columns

        # Define the transformation pipeline for numeric features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with the median
            ('scaler', StandardScaler())  # Scale numeric features to standardize their distribution
        ])

        # Define the transformation pipeline for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
        ])

        # Combine the numeric and categorical transformations into a single ColumnTransformer
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),  # Apply numeric transformations
            ('cat', categorical_transformer, categorical_features)  # Apply categorical transformations
        ])

        print("Preprocessing setup complete.")
    
    def build_pipeline(self) -> None:
        """
        Build the machine learning pipeline by integrating the preprocessing steps and the model.

        The pipeline consists of:
        - **Preprocessing Step:** This applies the transformations defined in the `preprocess_data` method, including
          imputation, scaling, and encoding of features.
        - **Model Step:** This step involves fitting the selected machine learning algorithm (e.g., logistic regression, 
          random forest) to the processed data.

        Note: The pipeline ensures that all transformations are applied consistently during both training and prediction phases.

        Raises:
        - ValueError: If the specified algorithm is unsupported for the determined task type.
        """
        print("Building the ML pipeline...")
        
        # Select the appropriate model based on the task type and specified algorithm
        if self.task_type == 'classification':
            if self.algorithm.lower() == 'logistic_regression':
                self.model = LogisticRegression(random_state=self.random_state)  # Logistic regression model
            elif self.algorithm.lower() == 'decision_tree':
                self.model = DecisionTreeClassifier(random_state=self.random_state)  # Decision tree classifier
            elif self.algorithm.lower() == 'random_forest':
                self.model = RandomForestClassifier(random_state=self.random_state)  # Random forest classifier
            else:
                raise ValueError("Unsupported algorithm for classification.")
        elif self.task_type == 'regression':
            if self.algorithm.lower() == 'linear_regression':
                self.model = LinearRegression()  # Linear regression model
            elif self.algorithm.lower() == 'decision_tree':
                self.model = DecisionTreeRegressor(random_state=self.random_state)  # Decision tree regressor
            elif self.algorithm.lower() == 'random_forest':
                self.model = RandomForestRegressor(random_state=self.random_state)  # Random forest regressor
            else:
                raise ValueError("Unsupported algorithm for regression.")
        else:
            raise ValueError("Invalid task type. Cannot build pipeline.")

        # Create the complete pipeline by combining preprocessing and model steps
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),  # Apply preprocessing
            ('model', self.model)  # Apply the machine learning model
        ])

        print("Pipeline built successfully.")
    
    
    def train_and_evaluate(self) -> None:
        """
        Train the model and evaluate its performance using cross-validation.

        This method applies k-fold cross-validation to assess the model's performance. For classification tasks, 
        the evaluation metric is accuracy. For regression tasks, the metric is negative mean squared error (MSE).

        Cross-validation splits the dataset into k folds, where k is defined by the num_folds parameter. 
        Each fold is used as a validation set once while the other k-1 folds are used for training. 
        The process is repeated k times to compute average performance metrics.

        Edge case: For classification, if the minimum class count is less than the number of folds, num_folds is adjusted dynamically to avoid errors.

        Raises:
        - NotFittedError: If the pipeline is not built before calling this method.
        - ValueError: If the number of folds is less than 2.
        """
        if self.pipeline is None:
            raise NotFittedError("Pipeline is not built. Call `build_pipeline()` first.")

        print("Starting model training and evaluation...")
        
        # Determine the number of folds for cross-validation based on the dataset and task type
        if self.task_type == 'classification':
            min_class_count = self.y.value_counts().min()  # Minimum class count for classification
            num_folds = min(self.num_folds, min_class_count, self.X.shape[0])
            if num_folds <2:
                num_folds = 2  # Ensure valid fold count
        else:
            num_folds = min(self.num_folds, self.X.shape[0])  # For regression, use dataset size

        if num_folds < 2:
            raise ValueError("The number of folds must be at least 2.")

        # Select scoring metric based on task type
        scoring = 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'
        try:
            scores = cross_val_score(self.pipeline, self.X, self.y, cv=num_folds, scoring=scoring, n_jobs = -1)  # Perform cross-validation
        except Exception as e:
            raise ValueError(f"An error occurred during cross-validation: {e}")

        # Display the evaluation results
        if self.task_type == 'classification':
            print(f"Cross-validated accuracy: {np.mean(scores):.4f}")
        else:
            print(f"Cross-validated MSE: {-np.mean(scores):.4f}")

        print("Model training and evaluation complete.")
    
    def save_pipeline(self, output_path: str = 'model_pipeline.pkl') -> None:
        """
        Save the trained pipeline to a file for later use.

        Parameters:
        - output_path (str): Path where the pipeline will be saved. Defaults to 'model_pipeline.pkl'.

        Typical Use Cases:
        - Deploying the model: The saved pipeline can be loaded into a production environment for predictions.
        - Reproducibility: The saved file ensures consistent model performance across different environments.
        - Further analysis: The saved pipeline can be reloaded for additional validation or further training.

        The saved file is in pickle format (.pkl), which can be loaded using joblib's `load` method.
        Note: Ensure compatibility of Python and library versions when loading the pipeline in a different environment.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline is not built. Call `build_pipeline()` first.")
        
        print(f"Saving the pipeline to {output_path}...")
        try:
            joblib.dump(self.pipeline, output_path)  # Save the pipeline using joblib
            print(f"Pipeline saved to {output_path}")  # Confirm the pipeline has been saved
        except Exception as e:
            raise ValueError(f"Failed to save the pipeline: {e}")

def main() -> None:
    """
    Main function to parse command-line arguments and execute the machine learning pipeline.

    Usage:
    ------
    This script can be executed from the command line with the following format:

        python script_name.py <csv_path> <target_column> <algorithm> [--random_state <int>] [--num_folds <int>] [--output_path <str>]

    Example:
    --------
    To run the script with a dataset:

        python app.py data.csv target_column random_forest --random_state 42 --num_folds 5 --output_path model_pipeline.pkl

    Arguments:
    ----------
    - csv_path (str): Path to the CSV file containing the dataset.
    - target_column (str): Name of the target column.
    - algorithm (str): Algorithm to use (e.g., 'random_forest').
    - random_state (int, optional): Random state for reproducibility. Defaults to 42.
    - num_folds (int, optional): Number of cross-validation folds. Defaults to 5.
    - output_path (str, optional): Path to save the trained pipeline. Defaults to 'model_pipeline.pkl'.
    """
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Machine Learning Pipeline Application')
    parser.add_argument('csv_path', type=str, help='Path to the CSV data file')  # Path to the input CSV file
    parser.add_argument('target_column', type=str, help='Name of the target column')  # Target column in the dataset
    parser.add_argument('algorithm', type=str, help='Algorithm to use')  # Machine learning algorithm to use
    parser.add_argument('--random_state', type=int, default=42, help='Random state')  # Random seed for reproducibility
    parser.add_argument('--num_folds', type=int, default=5, help='Number of cross-validation folds')  # Number of folds for CV
    parser.add_argument('--output_path', type=str, default='model_pipeline.pkl', help='Output path for the pipeline')  # File path to save pipeline

    args = parser.parse_args()  # Parse the arguments provided by the user

    # Create an instance of MLApplication with the parsed arguments
    app = MLApplication(csv_path=args.csv_path,
                        target_column=args.target_column,
                        algorithm=args.algorithm,
                        random_state=args.random_state,
                        num_folds=args.num_folds)
    try:
        app.load_data()  # Load the dataset
        app.preprocess_data()  # Preprocess the data
        app.build_pipeline()  # Build the ML pipeline
        app.train_and_evaluate()  # Train and evaluate the model
        app.save_pipeline(output_path=args.output_path)  # Save the trained pipeline
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except NotFittedError as nfe:
        print(f"NotFittedError: {nfe}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()  # Execute the main function
