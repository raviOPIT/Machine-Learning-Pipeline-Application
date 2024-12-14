# Application Architecture Documentation

## Overview

The `MLApplication` class is designed to automate and modularize the machine learning workflow. It streamlines tasks like data preprocessing, model selection, evaluation, and pipeline saving, enabling users to focus on building robust models without delving into repetitive setup procedures. This architecture ensures scalability, flexibility, and ease of use for both beginners and experienced practitioners.

## Core Components

### 1. `MLApplication` Class

The core component of the application, encapsulating end-to-end machine learning workflows. It includes attributes and methods that address each stage of the machine learning pipeline.

#### **Attributes**

1. **`csv_path` (str):**
   - Path to the CSV dataset file.
   - Example: `data/sample_data.csv`.

2. **`target_column` (str):**
   - The name of the target column for prediction.

3. **`algorithm` (str):**
   - The chosen machine learning algorithm.
   - Options:
     - Classification: `logistic_regression`, `decision_tree`, `random_forest`.
     - Regression: `linear_regression`, `decision_tree`, `random_forest`.

4. **`task_type` (str):**
   - Automatically inferred as `classification` or `regression` based on the target column's data type.

5. **`random_state` (int):**
   - A seed value for reproducibility.
   - Default: `42`.

6. **`num_folds` (int):**
   - Number of folds for cross-validation.
   - Default: `5`.

7. **`X` (DataFrame) and `y` (Series):**
   - `X`: Features extracted from the dataset.
   - `y`: Target variable extracted from the dataset.

8. **`preprocessor` (ColumnTransformer):**
   - Defines preprocessing steps for numeric and categorical columns.

9. **`pipeline` (Pipeline):**
   - A scikit-learn pipeline combining preprocessing and the machine learning model.

10. **`model` (Any):**
   - Stores the selected machine learning model.

#### **Methods**

1. **`load_data()`**
   - Loads data from the provided CSV file path.
   - Splits data into features (`X`) and target (`y`).
   - Automatically determines task type based on the target column’s data type.
   - **Error Handling:**
     - Missing files: Raises a `FileNotFoundError` with a descriptive message.
     - Empty datasets: Raises a `ValueError` if the dataset is empty.
     - Unsupported target data types: Raises a `ValueError` for invalid target column types.
     - Missing target column: Raises a `KeyError` with available column names.

2. **`preprocess_data()`**
   - Configures preprocessing for numeric and categorical columns:
     - Numeric features: Median imputation and standard scaling.
     - Categorical features: Most frequent value imputation and one-hot encoding.
   - Combines these transformations into a `ColumnTransformer`.
   - **Error Handling:**
     - Columns with unsupported data types: Raises a `ValueError` specifying problematic columns.
     - Columns containing only NaN values: Raises a `ValueError` for unprocessable columns.

3. **`build_pipeline()`**
   - Creates a scikit-learn pipeline by combining preprocessing steps with the selected model.
   - Dynamically supports algorithms based on the task type.
   - **Error Handling:**
     - Unsupported algorithms: Raises a `ValueError` for incompatible algorithm-task combinations.
     - Missing preprocessing steps: Ensures `preprocessor` is correctly initialized before proceeding.

4. **`train_and_evaluate()`**
   - Trains the pipeline and evaluates it using k-fold cross-validation.
   - Metrics:
     - Classification: Accuracy.
     - Regression: Negative Mean Squared Error (MSE).
   - Dynamically adjusts cross-validation folds based on dataset size or class distribution.
   - **Error Handling:**
     - Unbuilt pipelines: Raises a `NotFittedError` if `pipeline` is not initialized.
     - Invalid cross-validation folds: Dynamically adjusts or raises a `ValueError` for insufficient data.
     - Cross-validation errors: Catches and raises descriptive errors for unexpected issues.

5. **`save_pipeline(output_path)`**
   - Saves the trained pipeline to a file for reuse or deployment.
   - **Error Handling:**
     - Missing pipeline: Raises a `ValueError` if no pipeline is available for saving.
     - File write permissions: Raises an exception with a descriptive message.

---

## Command-Line Interface (CLI)

The application provides a CLI interface for quick and efficient execution.

### CLI Arguments

1. **Required:**
   - `csv_path`: Path to the dataset.
   - `target_column`: Target variable.
   - `algorithm`: Chosen model algorithm.

2. **Optional:**
   - `--random_state`: Random seed for reproducibility. Default: `42`.
   - `--num_folds`: Cross-validation folds. Default: `5`.
   - `--output_path`: Path to save the trained pipeline. Default: `model_pipeline.pkl`.

### Example

```bash
python app.py data/housing.csv price linear_regression --random_state 42 --num_folds 10 --output_path housing_pipeline.pkl
```

---

## Workflow Breakdown

1. **Initialization**
   - Instantiate the `MLApplication` class with user-specified parameters.

2. **Data Loading**
   - Reads the dataset and identifies the task type (classification or regression).
   - Includes robust error handling to validate dataset integrity.

3. **Preprocessing**
   - Configures transformations for numeric and categorical features.
   - Ensures compatibility of transformations with downstream tasks.
   - Raises detailed errors for unsupported or missing data.

4. **Pipeline Construction**
   - Builds a complete pipeline integrating preprocessing and modeling.
   - Validates compatibility between preprocessing and selected models.

5. **Training and Evaluation**
   - Performs k-fold cross-validation to evaluate model performance.
   - Adjusts cross-validation dynamically for small datasets or imbalanced classes.
   - Includes error handling for unexpected training or evaluation issues.

6. **Pipeline Saving**
   - Exports the pipeline as a `.pkl` file for reuse.
   - Includes error handling for file write permissions and disk space issues.

---

## Extensibility

1. **Adding Algorithms**
   - Update the `build_pipeline()` method to include new models.
   - Example: Add support for SVM, Gradient Boosting, or Neural Networks.

2. **Custom Preprocessing**
   - Extend the `preprocess_data()` method to include additional preprocessing steps.
   - Example: Handle text data, create interaction terms, or apply dimensionality reduction techniques like PCA.

3. **Advanced Metrics**
   - Add custom scoring functions in `train_and_evaluate()` for specialized evaluation.
   - Example: Use metrics like F1-score, AUC-ROC for classification, or R-squared for regression.

4. **Enhanced Data Validation**
   - Integrate data profiling tools like `pandas-profiling` or `Great Expectations` to detect anomalies before preprocessing.

---

## Dependencies

- **Python Libraries:**
  - `pandas`, `numpy`: Data manipulation.
  - `argparse`: CLI implementation.
  - `joblib`: Saving/loading pipelines.
- **scikit-learn:**
  - Models: Logistic Regression, Random Forest, Decision Tree, Linear Regression.
  - Tools: `Pipeline`, `ColumnTransformer`, `cross_val_score`.

---

## Future Enhancements

1. **Hyperparameter Tuning**
   - Integration with frameworks like `GridSearchCV`, `Optuna`, or `Ray Tune` for optimized parameter search.

2. **Unsupervised Learning**
   - Support for clustering, dimensionality reduction, or anomaly detection tasks.

3. **Visualization**
   - Generate visual reports of cross-validation results, feature importance, or preprocessing transformations.

4. **Cloud Integration**
   - Save pipelines to cloud storage (AWS S3, Google Cloud Storage) for collaborative workflows.
   - Enable model serving through cloud-based APIs.

5. **Logging and Monitoring**
   - Add logging capabilities using `logging` or `mlflow` for better traceability and debugging.

6. **Real-Time Predictions**
   - Extend functionality to enable batch and real-time predictions through REST APIs or message queues.

