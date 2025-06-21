# Class ModelEvaluator
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List, Dict, Union, Optional # Thêm Union, Optional



class ModelEvaluator:
    SUPPORTED_METRICS = ['mae', 'rmse', 'r2', 'mse'] # Thêm 'mse'

    def __init__(self, metrics_to_calculate: Optional[List[str]] = None):
        if metrics_to_calculate is None:
            self.metrics_to_calculate = ['mae', 'rmse', 'r2']
        else:
            if not isinstance(metrics_to_calculate, list) or not metrics_to_calculate:
                 raise ValueError("metrics_to_calculate phải là một list không rỗng các tên metric.")
            # Kiểm tra xem các metric yêu cầu có được hỗ trợ không
            invalid_metrics = [m for m in metrics_to_calculate if m not in self.SUPPORTED_METRICS]
            if invalid_metrics:
                raise ValueError(f"Các metric sau không được hỗ trợ: {invalid_metrics}. "
                                 f"Hỗ trợ: {self.SUPPORTED_METRICS}")
            self.metrics_to_calculate = list(set(metrics_to_calculate)) # Đảm bảo duy nhất

        print(f"ModelEvaluator initialized to calculate: {self.metrics_to_calculate}")


    def _calculate_mae(self, y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> float:
        """(Private) Tính Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)

    def _calculate_mse(self, y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> float:
        """(Private) Tính Mean Squared Error."""
        return mean_squared_error(y_true, y_pred)

    def _calculate_rmse(self, y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> float:
        """(Private) Tính Root Mean Squared Error."""
        # Có thể tính từ MSE để tránh tính lại bình phương
        # return np.sqrt(mean_squared_error(y_true, y_pred))
        mse = self._calculate_mse(y_true, y_pred)
        return np.sqrt(mse)


    def _calculate_r2(self, y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> float:
        """(Private) Tính R-squared score."""
        return r2_score(y_true, y_pred)

    def calculate_metrics(self, y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> Dict[str, float]:
        if len(y_true) != len(y_pred):
            print(f"Error: Length mismatch between y_true ({len(y_true)}) and y_pred ({len(y_pred)}).")
            return {}
        if len(y_true) == 0:
             print("Warning: Input arrays are empty. Returning empty metrics dictionary.")
             return {}

        results = {}
        calculation_map = {
            'mae': self._calculate_mae,
            'mse': self._calculate_mse,
            'rmse': self._calculate_rmse,
            'r2': self._calculate_r2
        }

        # print("\nCalculating specified metrics...") # Có thể bỏ log này để đỡ nhiễu
        for metric in self.metrics_to_calculate:
            calculation_func = calculation_map.get(metric)
            if calculation_func:
                try:
                    results[metric] = calculation_func(y_true, y_pred)
                    # print(f"  {metric.upper()}: {results[metric]:.4f}") # Log chi tiết nếu cần
                except Exception as e:
                    print(f"  Error calculating metric '{metric}': {e}")
                    results[metric] = np.nan # Hoặc giá trị khác để chỉ lỗi
            else:
                 # Trường hợp này không nên xảy ra do đã kiểm tra ở __init__
                 print(f"  Warning: Calculation function for metric '{metric}' not found.")

        return results
