# Class ModelTrainer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone, BaseEstimator # Thêm BaseEstimator
from typing import List, Dict, Optional, Any, Union
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C_kernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore') # Để tránh các cảnh báo không cần thiết trong ví dụ
# --- Giả sử các lớp trước đã được định nghĩa và có dữ liệu ---
# class ModelEvaluator: ... (Đã định nghĩa)
# df_selected_all: DataFrame từ FeatureSelector chứa features + targets + group
# final_feature_list: List[str] chứa tên các feature đã chọn

class ModelTrainer:
    SUPPORTED_MODELS = ['RandomForest', 'GradientBoosting', 'XGBoost','SVR','GaussianProcessRegressor']

    def __init__(self, model_type: str, model_params: Optional[Dict[str, Any]] = None):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model_type '{model_type}'. Supported: {self.SUPPORTED_MODELS}")

        self.model_type = model_type
        self.model_params = model_params if model_params is not None else self._get_default_params()
        self.trained_model: Optional[BaseEstimator] = None # Lưu trữ mô hình sklearn đã huấn luyện
        self.feature_names_in_: Optional[List[str]] = None # Lưu tên feature dùng để train

        print(f"ModelTrainer initialized for model type: {self.model_type} with params: {self.model_params}")

    def _get_default_params(self) -> Dict[str, Any]:
        """(Private) Trả về các tham số mặc định cho từng loại mô hình."""
        if self.model_type == 'RandomForest':
            return {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}
        elif self.model_type == 'GradientBoosting':
            return {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 1.0, 'random_state': 42}
        elif self.model_type == 'XGBoost':
            # Đảm bảo objective phù hợp cho hồi quy
            return {'objective': 'reg:squarederror', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1}
        elif self.model_type == 'SVR':
            return {'kernel':'rbf', 'C': 10, 'epsilon': 0.05, 'gamma': 'scale'}
        elif self.model_type == 'GaussianProcessRegressor':
            gpr_kernel_example = C_kernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            return {'kernel':gpr_kernel_example, 'random_state': 42, 'alpha': 1e-7, 'normalize_y': False}
        else:
            return {} # Trường hợp không nên xảy ra

    def _initialize_model(self) -> BaseEstimator:
        """(Private) Khởi tạo đối tượng mô hình dựa trên model_type."""
        if self.model_type == 'RandomForest':
            return RandomForestRegressor(**self.model_params)
        elif self.model_type == 'GradientBoosting':
            return GradientBoostingRegressor(**self.model_params)
        elif self.model_type == 'XGBoost':
            return xgb.XGBRegressor(**self.model_params)
        elif self.model_type == 'SVR':
            return SVR(**self.model_params)
        elif self.model_type == 'GaussianProcessRegressor':
            return GaussianProcessRegressor(**self.model_params)
        else:
             # Trường hợp này không nên xảy ra do đã kiểm tra ở __init__
             raise ValueError(f"Model type {self.model_type} not handled in _initialize_model")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'ModelTrainer':
        if not isinstance(X_train, pd.DataFrame) or X_train.empty:
            raise ValueError("X_train phải là một DataFrame không rỗng.")
        if not isinstance(y_train, pd.Series) or y_train.empty:
            raise ValueError("y_train phải là một Series không rỗng.")
        if len(X_train) != len(y_train):
            raise ValueError("Độ dài của X_train và y_train không khớp.")

        print(f"\nTraining {self.model_type} model...")
        self.trained_model = self._initialize_model()
        self.feature_names_in_ = X_train.columns.tolist() # Lưu lại tên feature

        try:
            self.trained_model.fit(X_train, y_train)
            print(f"{self.model_type} model training complete.")
        except Exception as e:
            print(f"Error during {self.model_type} model training: {e}")
            self.trained_model = None # Đặt lại nếu lỗi
            raise e # Ném lại lỗi để xử lý bên ngoài nếu cần

        return self

    def predict(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if self.trained_model is None:
            print("Error: Model has not been trained yet. Call train() first.")
            return None
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")

        # Kiểm tra và sắp xếp lại cột nếu cần (để đảm bảo thứ tự đúng)
        if self.feature_names_in_:
             if list(X.columns) != self.feature_names_in_:
                  print("Warning: Input feature columns order mismatch or different columns. Reordering based on training features.")
                  try:
                       X = X[self.feature_names_in_] # Sắp xếp lại theo đúng thứ tự
                  except KeyError as e:
                       raise ValueError(f"Input data is missing columns used during training: {e}")
        else:
             print("Warning: Feature names used during training not stored. Assuming input columns are in the correct order.")


        print(f"Predicting with trained {self.model_type} model...")
        try:
            predictions = self.trained_model.predict(X)
            return predictions
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

    def get_model(self) -> Optional[BaseEstimator]:
        """Trả về đối tượng mô hình đã huấn luyện."""
        return self.trained_model

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if self.trained_model is None or not hasattr(self.trained_model, 'feature_importances_'):
            print(f"Warning: Model is not trained or does not support feature_importances_ ({self.model_type}).")
            return None
        if self.feature_names_in_ is None:
            print("Warning: Feature names not stored during training. Cannot create importance DataFrame with names.")
            
            return None

        importances = self.trained_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.feature_names_in_,
            'Importance': importances
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return importance_df
