# MLApplication API Reference

## Overview

This document provides a detailed reference for the `MLApplication` class, including its attributes and methods. It describes the inputs, outputs, and functionality of each component, along with extensive error handling and usage scenarios.

---

## Class: `MLApplication`

### Constructor

#### `__init__(csv_path: str, target_column: str, algorithm: str, random_state: Optional[int] = 42, num_folds: Optional[int] = 5)`

**Description:** Initializes the `MLApplication` object with specified parameters.

**Parameters:**
- `csv_path` (str): Path to the CSV file containing the dataset.
- `target_column` (str): Name of the target column in the dataset.
- `algorithm` (str): Name of the machine learning algorithm to use (e.g., `logistic_regression`, `decision_tree`).
- `random_state` (Optional[int]): Random seed for reproducibility. Default is `42`.
- `num_folds` (Optional[int]): Number of cross-validation folds. Default is `5`.

---

### Methods

#### 1. `load_data()`

**Description:**
Loads the dataset from the specified CSV file and separates features (`X`) and target (`y`). Automatically determines the task type (`classification` or `regression`) based on the target column's data type.

**Inputs:** None (uses `self.csv_path` and `self.target_column`).

**Outputs:**
- `self.X` (DataFrame): Independent variables.
- `self.y` (Series): Dependent variable.
- `self.task_type` (str): Automatically determined as `classification` or `regression`.

**Error Handling:**
- Raises `FileNotFoundError` if the file does not exist.
- Raises `KeyError` if the `target_column` is not found.
- Raises `ValueError` for unsupported or missing target column types.

**Additional Notes:**
- Handles cases where the dataset is empty or contains missing columns.
- Logs the identified task type for transparency.

---

#### 2. `preprocess_data()`

**Description:**
Configures preprocessing pipelines for numeric and categorical data using `ColumnTransformer`. Handles missing values, scales numeric features, and encodes categorical features.

**Inputs:** None (operates on `self.X` and `self.y` ).

**Outputs:**
- `self.preprocessor` (ColumnTransformer): Preprocessing pipeline.

**Error Handling:**
- Raises `ValueError` if columns contain unsupported data types.
- Raises `ValueError` if columns or target contain only NaN values.

**Additional Notes:**
- Automatically identifies numeric and categorical features.
- Ensures all transformations are robust to edge cases.

---

#### 3. `build_pipeline()`

**Description:**
Constructs a complete machine learning pipeline by combining the preprocessor and the selected model.

**Inputs:** None (uses `self.algorithm` and `self.task_type`).

**Outputs:**
- `self.pipeline` (Pipeline): Pipeline including preprocessing and the model.

**Error Handling:**
- Raises `ValueError` for unsupported algorithms or task types.

**Additional Notes:**
- Supports dynamic addition of preprocessing steps or model components.
- Validates compatibility of selected model with the identified task type.

---

#### 4. `train_and_evaluate()`

**Description:**
Trains the pipeline using cross-validation and evaluates performance metrics.

**Inputs:** None (uses `self.pipeline`, `self.X`, `self.y`, and `self.num_folds`).

**Outputs:**
- Prints evaluation metrics:
  - Classification: Cross-validated accuracy.
  - Regression: Cross-validated mean squared error (MSE).

**Error Handling:**
- Raises `NotFittedError` if the pipeline is not built.
- Dynamically adjusts cross-validation folds if dataset size is insufficient.
- Raises `ValueError` for cross-validation errors.

**Additional Notes:**
- Automatically handles imbalanced datasets by adjusting fold counts.


---

#### 5. `save_pipeline(output_path: str = 'model_pipeline.pkl')`

**Description:**
Saves the constructed pipeline to a specified file using `joblib`.

**Inputs:**
- `output_path` (str): Path to save the pipeline. Default is `'model_pipeline.pkl'`.

**Outputs:**
- Saves the pipeline to the specified path.

**Error Handling:**
- Raises `ValueError` if no pipeline is available for saving.
- Handles file write permission issues gracefully.

**Additional Notes:**
- Ensures compatibility of saved pipelines across different environments.
- Logs the save location for future reference.

---

## Usage Example

```python
from app import MLApplication

# Initialize the application
app = MLApplication(
    csv_path="data.csv",
    target_column="target",
    algorithm="logistic_regression",
    random_state=42,
    num_folds=5
)

# Load data
app.load_data()

# Preprocess data
app.preprocess_data()

# Build pipeline
app.build_pipeline()

# Train and evaluate
app.train_and_evaluate()

# Save pipeline
app.save_pipeline(output_path="model_pipeline.pkl")
```

---

## Dependencies

- `pandas`: For data manipulation.
- `numpy`: For numerical operations.
- `joblib`: For saving pipelines.
- `scikit-learn`: For preprocessing, modeling, and evaluation.
- `argparse`: For command-line argument parsing.

**Additional Details:**
- Compatible with Python 3.7 and above.
- Modular design allows easy integration with other machine learning libraries.

---

## Notes

- The application supports classification and regression tasks.
- It is designed to be modular, allowing extensions for additional algorithms or preprocessing steps.
- Error handling ensures that the user receives informative messages for invalid inputs or operations.
- Logs provide detailed insights into workflow execution, facilitating debugging and transparency.

