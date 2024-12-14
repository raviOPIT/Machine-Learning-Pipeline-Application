# Usage Documentation

## Command-Line Interface (CLI) Usage

This application offers a command-line interface to execute machine learning pipelines seamlessly. Below are detailed usage instructions.

### Command Format

```bash
python app.py CSV_PATH TARGET_COLUMN ALGORITHM [OPTIONS]
```

### Required Arguments

1. **`CSV_PATH`**: Path to the input CSV file.
2. **`TARGET_COLUMN`**: Name of the target column in the dataset.
3. **`ALGORITHM`**: Specify the algorithm to use:
   - For classification: `logistic_regression`, `decision_tree`, `random_forest`.
   - For regression: `linear_regression`, `decision_tree`, `random_forest`.

### Optional Arguments

1. **`--random_state`**: Seed for reproducibility. Defaults to 42.
2. **`--num_folds`**: Number of cross-validation folds. Defaults to 5.
3. **`--output_path`**: Path to save the trained pipeline. Defaults to `model_pipeline.pkl`.

### Example Commands

#### Example 1: Classification Task

```bash
python app.py data/iris.csv species random_forest --random_state 42 --num_folds 5 --output_path iris_pipeline.pkl
```

**Explanation:**
- Uses the `iris.csv` dataset.
- Predicts the `species` column using a `random_forest` algorithm.
- Performs 5-fold cross-validation.
- Saves the trained pipeline to `iris_pipeline.pkl`.

#### Example 2: Regression Task

```bash
python app.py data/housing.csv price linear_regression --random_state 42 --num_folds 10 --output_path housing_pipeline.pkl
```

**Explanation:**
- Uses the `housing.csv` dataset.
- Predicts the `price` column using a `linear_regression` algorithm.
- Performs 10-fold cross-validation.
- Saves the trained pipeline to `housing_pipeline.pkl`.

---

## Jupyter Notebook Usage

This application can be used interactively in Jupyter Notebooks. Below is a step-by-step guide.

### Step 1: Import the Class

```python
from app import MLApplication
```

### Step 2: Initialize the Application

Create an instance of the `MLApplication` class with appropriate parameters:

```python
app = MLApplication(
    csv_path='data/iris.csv',
    target_column='species',
    algorithm='random_forest',
    random_state=42,
    num_folds=5
)
```

### Step 3: Load Data

```python
app.load_data()
```
This loads the dataset and determines whether the task is classification or regression based on the target column.

### Step 4: Preprocess Data

```python
app.preprocess_data()
```
This handles missing values, scales numeric features, and one-hot encodes categorical features.

### Step 5: Build the Pipeline

```python
app.build_pipeline()
```
This combines preprocessing steps and the specified machine learning algorithm into a single pipeline.

### Step 6: Train and Evaluate

```python
app.train_and_evaluate()
```
Performs cross-validation and outputs metrics (accuracy for classification or MSE for regression).

### Step 7: Save the Pipeline

```python
app.save_pipeline(output_path='trained_pipeline.pkl')
```
Saves the trained pipeline for later use.

### Full Example

```python
from app import MLApplication

app = MLApplication(
    csv_path='data/housing.csv',
    target_column='price',
    algorithm='linear_regression',
    random_state=42,
    num_folds=10
)

app.load_data()
app.preprocess_data()
app.build_pipeline()
app.train_and_evaluate()
app.save_pipeline(output_path='housing_pipeline.pkl')
```

---

## Tips and Best Practices

1. **Data Quality:** Ensure the dataset is clean, with minimal missing or erroneous values.
2. **Target Column:** Double-check the target column's data type to avoid mismatches.
3. **Algorithm Selection:** Use appropriate algorithms for the task type (classification or regression).
4. **Cross-Validation Folds:** Higher folds improve metric stability but increase runtime.
5. **Reproducibility:** Always set a `random_state` for consistent results.

For additional examples, see the `example_notebook.ipynb` file in the repository.

