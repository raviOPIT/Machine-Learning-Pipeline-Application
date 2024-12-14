# Machine Learning Pipeline Experimentation Application

## Project Overview

This application provides a robust and user-friendly framework for building and experimenting with machine learning pipelines. It simplifies tasks like preprocessing, model training, and evaluation, making it suitable for developers, data scientists, and researchers. The application supports both classification and regression tasks and can be used via a Command-Line Interface (CLI) or within a Jupyter Notebook. The design adheres to Object-Oriented Programming (OOP) principles for modularity and scalability.

**Purpose and Functionality:**

- Automates the machine learning pipeline for classification and regression tasks.
- Detects whether the task is classification or regression based on the target column's data type.
- Handles data preprocessing, including imputation, scaling, and encoding.
- Supports cross-validation to ensure robust evaluation.
- Offers saving mechanisms for trained pipelines.
- Provides detailed error handling and feedback for seamless experimentation, ensuring graceful handling of issues such as missing files, invalid data types, or unsupported algorithms.

---

## Architecture Overview

The core of the application is the `MLApplication` class, which integrates all steps of a typical machine learning workflow:

- **Data Handling:**

  - Loads datasets from CSV files.
  - Automatically identifies task type (classification or regression) based on the target variable.

- **Preprocessing:**

  - Imputes missing values for numeric and categorical features.
  - Scales numeric features and one-hot encodes categorical features.

- **Pipeline Construction:**

  - Combines preprocessing steps with model training in a single pipeline.
  - Supports Logistic Regression, Decision Tree, and Random Forest for classification, and Linear Regression, Decision Tree Regressor, and Random Forest Regressor for regression tasks.

- **Model Training and Evaluation:**

  - Uses k-fold cross-validation for reliable performance metrics.
  - Provides detailed feedback on accuracy (classification) or Mean Squared Error (regression).

- **Pipeline Saving:**

  - Saves the trained pipeline as a `.pkl` file for future use.

---

## Usage Instructions

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have Python 3.7 or higher and the following libraries:

   - pandas
   - numpy
   - scikit-learn
   - joblib
   - pytest (for testing)

### Running the Application from the Command Line

1. **Basic Command Structure:**

   ```bash
   python app.py CSV_PATH TARGET_COLUMN ALGORITHM [OPTIONS]
   ```

2. **Required Arguments:**

   - `CSV_PATH`: Path to the input CSV file.
   - `TARGET_COLUMN`: Name of the target column.
   - `ALGORITHM`: Algorithm to use (e.g., logistic_regression, random_forest).

3. **Optional Arguments:**

   - `--random_state`: Random seed (default: 42).
   - `--num_folds`: Number of cross-validation folds (default: 5).
   - `--output_path`: Path to save the trained pipeline (default: model_pipeline.pkl).

4. **Example Command:**

   ```bash
   python app.py housing.csv price random_forest --random_state 42 --num_folds 5 --output_path pipeline.pkl
   ```

### Using the Application in a Jupyter Notebook

1. **Load and Initialize the Application:**

   ```python
   from app import MLApplication

   app = MLApplication(
       csv_path='example.csv',
       target_column='Transported',
       algorithm='random_forest',
       random_state=42,
       num_folds=5
   )
   ```

2. **Execute the Pipeline Steps:**

   ```python
   app.load_data()
   app.preprocess_data()
   app.build_pipeline()
   app.train_and_evaluate()
   app.save_pipeline(output_path='trained_pipeline.pkl')
   ```

3. **Example Notebook:** Refer to `example_notebook.ipynb` for a comprehensive walkthrough with comments and markdown.

---
## Tests

The application has been rigorously tested with a comprehensive suite of tests to ensure robustness and reliability. Below is a detailed breakdown of the test categories and corresponding test cases found in the `tests/test_app.py` file:

### Test Categories

#### Initialization Tests:
- **`test_initialization`**: Verifies correct initialization with default parameters.

#### Data Loading Tests:
- **`test_load_data`**: Tests data loading for classification tasks.
- **`test_load_data_regression`**: Tests data loading for regression tasks.
- **`test_load_data_multiclass`**: Tests data loading for multiclass classification.
- **`test_load_data_empty_dataset`**: Checks if loading an empty dataset raises a `ValueError`.
- **`test_load_data_missing_file`**: Ensures `FileNotFoundError` is raised for nonexistent files.
- **`test_load_data_missing_target_column`**: Verifies `KeyError` is raised if the target column is missing.

#### Data Preprocessing Tests:
- **`test_preprocess_data`**: Ensures preprocessing pipeline creation for classification data.
- **`test_preprocess_data_all_nan_features`**: Verifies `ValueError` if a feature column contains only `NaN` values.
- **`test_preprocess_data_invalid_data_types`**: Ensures `ValueError` for unsupported feature data types.
- **`test_preprocess_data_multiclass`**: Validates preprocessing for multiclass classification data.

#### Pipeline Building Tests:
- **`test_build_pipeline_valid_algorithm`**: Validates pipeline building with a supported algorithm.
- **`test_build_pipeline_invalid_algorithm`**: Ensures `ValueError` for unsupported algorithms.

#### Model Training and Evaluation Tests:
- **`test_train_and_evaluate_classification`**: Verifies training and evaluation for classification tasks.
- **`test_train_and_evaluate_regression`**: Verifies training and evaluation for regression tasks.
- **`test_train_and_evaluate_multiclass`**: Verifies training and evaluation for multiclass classification tasks.
- **`test_train_and_evaluate_imbalanced_classes`**: Ensures the pipeline handles imbalanced classes during training.
- **`test_train_and_evaluate_without_pipeline`**: Ensures `NotFittedError` if the pipeline is not built.

#### Pipeline Saving Tests:
- **`test_save_pipeline`**: Verifies the pipeline is saved correctly.
- **`test_save_pipeline_failure`**: Ensures `ValueError` if attempting to save a pipeline before building it.

#### Edge Case Tests:
- **`test_single_row_dataset`**: Verifies training fails with a single-row dataset.
- **`test_large_dataset`**: Verifies the pipeline handles large datasets efficiently.
- **`test_all_nan_target_column`**: Validates preprocessing fails if the target column contains only `NaN` values.

#### Integration Tests:
- **`test_end_to_end_pipeline`**: Validates the complete pipeline for classification tasks.
- **`test_end_to_end_pipeline_regression`**: Validates the complete pipeline for regression tasks.
- **`test_end_to_end_pipeline_multiclass`**: Validates the complete pipeline for multiclass classification tasks.

### Running Tests

To execute the tests, use the following command:

```bash
pytest tests/
```

This will execute all test cases and provide a summary of the results. Passed tests are indicated with a green `.` or `PASSED` message, while failed tests are shown with a red `F` or `FAILED` message, including the error traceback.



## Developer Instructions

### Running Unit Tests

1. **Command to Run Tests:**

   ```bash
   pytest tests/
   ```

2. **Interpreting Test Results:**

   - Passed tests are indicated with a green `.` or `PASSED` message.
   - Failed tests are shown with a red `F` or `FAILED` message, including the error traceback.
   - Example:
     ```
     ============================= test session starts =============================
     collected 26 items

     tests/test_app.py .........F                                      [ 90%]
     tests/test_app.py::test_invalid_algorithm FAILED                 [ 10%]

     ============================= FAILURES =============================
     ```

3. **Expanded Test Coverage:**

   - **Initialization Tests:**
     - Ensure correct setup of `MLApplication` with expected parameters.
   - **Data Handling Tests:**
     - Validate loading of datasets with varying configurations (e.g., empty datasets, missing target columns).
     - Test task type detection for classification, regression, and multiclass classification.
   - **Preprocessing Tests:**
     - Check transformations for numeric and categorical features, including missing values and unsupported data types.
     - Validate preprocessing for edge cases like all-NaN columns or invalid data types.
   - **Pipeline Tests:**
     - Verify pipeline creation for supported algorithms and ensure errors for unsupported ones.
   - **Model Training and Evaluation Tests:**
     - Ensure metrics like accuracy and MSE are calculated correctly.
     - Validate handling of imbalanced classes, small datasets, and multiclass problems.
   - **Pipeline Saving Tests:**
     - Confirm successful saving and reloading of pipelines for reuse.
   - **Integration Tests:**
     - Validate end-to-end pipeline execution, from loading data to saving trained models.

### Debugging and Troubleshooting

**Common Issues and Solutions:**

1. **File Not Found:**

   - Ensure the `CSV_PATH` is correct and points to an existing file.

2. **Invalid Algorithm:**

   - Use only supported algorithms (e.g., logistic_regression, random_forest).

3. **Empty or Corrupt Dataset:**

   - Verify the dataset has valid data and the target column is not missing or empty.
   - Ensure that the target column name does not have white spaces. if there are white spaces, replace the space with an underscore "_" in the .csv file.
   - For classification tasks, ensure that the target column datatype is categorical.
   

4. **Preprocessing Failures:**

   - Check that numeric and categorical columns are correctly formatted.
   - Ensure no columns contain only `NaN` values.

5. **Cross-Validation Errors:**

   - Ensure the dataset size supports the specified number of folds.

6. **Pipeline Build or Training Issues:**
   - Verify all required steps (`load_data`, `preprocess_data`, `build_pipeline`) have been completed before training.
   - Check for errors related to incompatible data types or features.

7. **Pipeline Saving Errors:**
   - Ensure the pipeline is built and trained before saving.
   - Verify the specified output path is writable.

---

## Additional Documentation

- docs/usage.md: Detailed usage instructions with advanced parameters.
- docs/architecture.md: Explanation of the application’s architecture and design decisions.
- docs/api_reference.md: Documentation of classes and methods.

---


## Contact

For questions or issues, open an issue on the repository or contact the developer directly.

