import pytest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from app import MLApplication
from joblib import load

# =======================
# Fixtures
# =======================

@pytest.fixture
def sample_classification_data(tmp_path):
    """
    Creates a sample classification CSV file for testing.
    
    The dataset includes numeric and categorical features along with a binary target column.
    Missing values are included to test imputation logic.

    Returns:
        str: The file path of the created CSV.
    """
    data = {
        "feature1": [1.0, 2.0, 3.0, np.nan],  # Numeric feature with a missing value
        "feature2": ["A", "B", "C", "A"],  # Categorical feature
        "target": [True, False, True, False]  # Binary classification target
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "classification_sample.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def sample_regression_data(tmp_path):
    """
    Creates a sample regression CSV file for testing.

    The dataset includes numeric and categorical features along with a continuous target column.
    Missing values are included to test imputation logic.

    Returns:
        str: The file path of the created CSV.
    """
    data = {
        "feature1": [1.0, 2.0, 3.0, np.nan],  # Numeric feature with a missing value
        "feature2": ["A", "B", "C", "A"],  # Categorical feature
        "target": [10.5, 20.3, 30.7, 15.2]  # Continuous target for regression
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "regression_sample.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def sample_multiclass_classification_data(tmp_path):
    """
    Creates a sample multiclass classification CSV file for testing.

    Returns:
        str: The file path of the created CSV.
    """
    data = {
        "feature1": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0],
        "feature2": ["A", "B", "C", "D", "A", "C", "D"],
        "target": ["X", "Y", "Z", "Z", "X", "Y", "Y"]  # Multiclass target
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "multiclass_sample.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def classification_app_instance(sample_classification_data):
    """
    Creates an instance of MLApplication for classification tasks.

    Returns:
        MLApplication: The initialized MLApplication instance.
    """
    return MLApplication(
        csv_path=sample_classification_data,
        target_column="target",
        algorithm="random_forest"
    )

@pytest.fixture
def regression_app_instance(sample_regression_data):
    """
    Creates an instance of MLApplication for regression tasks.

    Returns:
        MLApplication: The initialized MLApplication instance.
    """
    return MLApplication(
        csv_path=sample_regression_data,
        target_column="target",
        algorithm="random_forest"
    )

@pytest.fixture
def multiclass_app_instance(sample_multiclass_classification_data):
    """
    Creates an instance of MLApplication for multiclass classification tasks.

    Returns:
        MLApplication: The initialized MLApplication instance.
    """
    return MLApplication(
        csv_path=sample_multiclass_classification_data,
        target_column="target",
        algorithm="random_forest"
    )

# =======================
# Tests for Initialization
# =======================

def test_initialization(classification_app_instance):
    """
    Verifies that the MLApplication instance is initialized correctly with default parameters.
    """
    assert classification_app_instance.csv_path.endswith("classification_sample.csv")
    assert classification_app_instance.target_column == "target"
    assert classification_app_instance.algorithm == "random_forest"
    assert classification_app_instance.random_state == 42
    assert classification_app_instance.num_folds == 5

# =======================
# Tests for load_data
# =======================

def test_load_data(classification_app_instance):
    """
    Ensures that data is loaded correctly for a classification task and task type is identified as 'classification'.
    """
    classification_app_instance.load_data()
    assert classification_app_instance.X is not None
    assert classification_app_instance.y is not None
    assert classification_app_instance.task_type == "classification"

def test_load_data_regression(regression_app_instance):
    """
    Ensures that data is loaded correctly for a regression task and task type is identified as 'regression'.
    """
    regression_app_instance.load_data()
    assert regression_app_instance.task_type == "regression"

def test_load_data_multiclass(multiclass_app_instance):
    """
    Ensures that data is loaded correctly for a multiclass classification task.
    """
    multiclass_app_instance.load_data()
    assert multiclass_app_instance.task_type == "classification"

def test_load_data_empty_dataset(tmp_path):
    """
    Validates that loading an empty dataset raises a ValueError.
    """
    empty_file = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(empty_file, index=False)
    app = MLApplication(
        csv_path=str(empty_file),
        target_column="target",
        algorithm="random_forest"
    )
    with pytest.raises(ValueError, match="The dataset is empty"):
        app.load_data()

def test_load_data_missing_file():
    """
    Ensures FileNotFoundError is raised for nonexistent files.
    """
    with pytest.raises(FileNotFoundError):
        app = MLApplication(
            csv_path="nonexistent.csv",
            target_column="target",
            algorithm="logistic_regression"
        )
        app.load_data()

def test_load_data_missing_target_column(sample_classification_data):
    """
    Validates that a KeyError is raised if the specified target column is missing in the dataset.
    """
    app = MLApplication(
        csv_path=sample_classification_data,
        target_column="nonexistent_column",
        algorithm="random_forest"
    )
    with pytest.raises(KeyError, match="Column 'nonexistent_column' does not exist"):
        app.load_data()

# =======================
# Tests for preprocess_data
# =======================

def test_preprocess_data(classification_app_instance):
    """
    Ensures that the preprocessing pipeline is created successfully for classification data.
    """
    classification_app_instance.load_data()
    classification_app_instance.preprocess_data()
    assert classification_app_instance.preprocessor is not None

def test_preprocess_data_all_nan_features(classification_app_instance):
    """
    Verifies that ValueError is raised if a feature column contains only NaN values.
    """
    classification_app_instance.load_data()
    classification_app_instance.X["feature1"] = np.nan
    with pytest.raises(ValueError, match="contain only NaN values"):
        classification_app_instance.preprocess_data()

def test_preprocess_data_invalid_data_types(classification_app_instance):
    """
    Ensures that ValueError is raised for unsupported feature data types.
    """
    classification_app_instance.load_data()
    classification_app_instance.X["feature1"] = pd.to_datetime(classification_app_instance.X["feature1"], errors='coerce')
    with pytest.raises(ValueError, match="Unsupported data types found in features"):
        classification_app_instance.preprocess_data()

def test_preprocess_data_multiclass(multiclass_app_instance):
    """
    Validates preprocessing for multiclass classification data.
    """
    multiclass_app_instance.load_data()
    multiclass_app_instance.preprocess_data()
    assert multiclass_app_instance.preprocessor is not None

# =======================
# Tests for build_pipeline
# =======================

def test_build_pipeline_valid_algorithm(classification_app_instance):
    """
    Validates that the pipeline is built successfully with a supported algorithm.
    """
    classification_app_instance.load_data()
    classification_app_instance.preprocess_data()
    classification_app_instance.build_pipeline()
    assert classification_app_instance.pipeline is not None

def test_build_pipeline_invalid_algorithm(classification_app_instance):
    """
    Ensures that ValueError is raised for unsupported algorithms.
    """
    classification_app_instance.algorithm = "unsupported_algo"
    with pytest.raises(ValueError):
        classification_app_instance.build_pipeline()


# =======================
# Tests for train_and_evaluate
# =======================

def test_train_and_evaluate_classification(classification_app_instance):
    """
    Verifies successful training and evaluation for a classification task.
    """
    classification_app_instance.load_data()
    classification_app_instance.preprocess_data()
    classification_app_instance.build_pipeline()
    classification_app_instance.train_and_evaluate()

def test_train_and_evaluate_regression(regression_app_instance):
    """
    Verifies successful training and evaluation for a regression task.
    """
    regression_app_instance.load_data()
    regression_app_instance.preprocess_data()
    regression_app_instance.build_pipeline()
    regression_app_instance.train_and_evaluate()

def test_train_and_evaluate_multiclass(multiclass_app_instance):
    """
    Verifies successful training and evaluation for a multiclass classification task.
    """
    multiclass_app_instance.load_data()
    multiclass_app_instance.preprocess_data()
    multiclass_app_instance.build_pipeline()
    multiclass_app_instance.train_and_evaluate()

def test_train_and_evaluate_imbalanced_classes(tmp_path):
    """
    Ensures that the pipeline handles imbalanced classes without errors during training.
    """
    data = {
        "feature1": [1.0, 2.0, 3.0, 4.0],
        "feature2": ["A", "B", "A", "B"],
        "target": [0, 0, 0, 1]  # Imbalanced classes
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "imbalanced_classes.csv"
    df.to_csv(file_path, index=False)
    app = MLApplication(
        csv_path=str(file_path),
        target_column="target",
        algorithm="random_forest"
    )
    app.load_data()
    app.preprocess_data()
    app.build_pipeline()
    app.train_and_evaluate()

def test_train_and_evaluate_without_pipeline(classification_app_instance):
    """
    Ensures NotFittedError is raised if the pipeline is not built.
    """
    classification_app_instance.load_data()
    with pytest.raises(NotFittedError, match="Pipeline is not built."):
        classification_app_instance.train_and_evaluate()

# =======================
# Tests for save_pipeline
# =======================

def test_save_pipeline(classification_app_instance, tmp_path):
    """
    Verifies that the pipeline is saved correctly to the specified path.
    """
    output_path = tmp_path / "pipeline.pkl"
    classification_app_instance.load_data()
    classification_app_instance.preprocess_data()
    classification_app_instance.build_pipeline()
    classification_app_instance.save_pipeline(output_path=str(output_path))
    assert output_path.exists()

def test_save_pipeline_failure(tmp_path):
    """
    Ensures that attempting to save a pipeline before building it raises a ValueError.
    """
    app = MLApplication(
        csv_path="nonexistent.csv",
        target_column="target",
        algorithm="random_forest"
    )
    with pytest.raises(ValueError, match="Pipeline is not built"):
        app.save_pipeline(output_path="/invalid/path")



# =======================
# Edge Case Tests
# =======================

def test_single_row_dataset(tmp_path):
    """
    Verifies that training fails with a single-row dataset due to insufficient data for cross-validation.
    """
    data = {"feature1": [1.0], "feature2": ["A"], "target": [0]}
    df = pd.DataFrame(data)
    file_path = tmp_path / "single_row.csv"
    df.to_csv(file_path, index=False)
    app = MLApplication(
        csv_path=str(file_path),
        target_column="target",
        algorithm="random_forest"
    )
    app.load_data()
    app.preprocess_data()
    app.build_pipeline()
    with pytest.raises(ValueError):
        app.train_and_evaluate()

def test_large_dataset(tmp_path):
    """
    Verifies that the pipeline can handle large datasets efficiently without errors.
    """
    data = {
        "feature1": np.random.rand(10000),
        "feature2": np.random.choice(["A", "B", "C"], size=10000),
        "target": np.random.randint(0, 2, size=10000)
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "large_dataset.csv"
    df.to_csv(file_path, index=False)
    app = MLApplication(
        csv_path=str(file_path),
        target_column="target",
        algorithm="random_forest"
    )
    app.load_data()
    app.preprocess_data()
    app.build_pipeline()
    app.train_and_evaluate()

def test_all_nan_target_column(tmp_path):
    """
    Validates that preprocessing fails if the target column contains only NaN values.
    """
    data = {
            "feature1": [1.0, 2.0, 3.0, 4.0],
            "feature2": ["B", "B", "C", "D"],
            "target": [np.nan, np.nan, np.nan, np.nan]
        }
    df = pd.DataFrame(data)
    file_path = tmp_path / "all_nan_target.csv"
    df.to_csv(file_path, index=False)
    app = MLApplication(
                csv_path=str(file_path),
                target_column="target",
                algorithm="random_forest"
            )
    app.load_data()
    with pytest.raises(ValueError, match="The target column contains only NaN values"):
        app.preprocess_data()

# =======================
# Integration Tests
# =======================

def test_end_to_end_pipeline(classification_app_instance, tmp_path):
    """
    Validates the complete pipeline from data loading to saving for classification tasks.
    """
    output_path = tmp_path / "end_to_end_pipeline.pkl"
    classification_app_instance.load_data()
    classification_app_instance.preprocess_data()
    classification_app_instance.build_pipeline()
    classification_app_instance.train_and_evaluate()
    classification_app_instance.save_pipeline(output_path=str(output_path))
    assert output_path.exists()

def test_end_to_end_pipeline_regression(regression_app_instance, tmp_path):
    """
    Validates the complete pipeline from data loading to saving for regression tasks.
    """
    output_path = tmp_path / "end_to_end_regression_pipeline.pkl"
    regression_app_instance.load_data()
    regression_app_instance.preprocess_data()
    regression_app_instance.build_pipeline()
    regression_app_instance.train_and_evaluate()
    regression_app_instance.save_pipeline(output_path=str(output_path))
    assert output_path.exists()

def test_end_to_end_pipeline_multiclass(multiclass_app_instance, tmp_path):
    """
    Validates the complete pipeline from data loading to saving for multiclass classification tasks.
    """
    output_path = tmp_path / "end_to_end_multiclass_pipeline.pkl"
    multiclass_app_instance.load_data()
    multiclass_app_instance.preprocess_data()
    multiclass_app_instance.build_pipeline()
    multiclass_app_instance.train_and_evaluate()
    multiclass_app_instance.save_pipeline(output_path=str(output_path))
    assert output_path.exists()
