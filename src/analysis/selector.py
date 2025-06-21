# Class FeatureSelector
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Union # Thêm Union
from analyzer import FeatureAnalyzer


class FeatureSelector:
    SUPPORTED_METHODS = ['manual', 'correlation_threshold', 'all_numeric'] # Các phương pháp hỗ trợ

    def __init__(self,
                 combined_data: pd.DataFrame,
                 target_cols: Union[str, List[str]], # Target để tính tương quan hoặc loại trừ
                 group_col: Optional[str] = 'battery_id', # Cột định danh nhóm để loại trừ
                 other_cols_to_exclude: Optional[List[str]] = None # Các cột khác cần loại trừ (vd: 'cycle', 'capacity')
                ):
        if not isinstance(combined_data, pd.DataFrame) or combined_data.empty:
            raise ValueError("Input 'combined_data' phải là một DataFrame không rỗng.")
        self.data: pd.DataFrame = combined_data.copy()

        # Chuẩn hóa target_cols thành list
        if isinstance(target_cols, str):
            self.target_cols: List[str] = [target_cols]
        elif isinstance(target_cols, list):
            self.target_cols: List[str] = target_cols
        else:
            raise TypeError("target_cols phải là str hoặc list[str].")

        self.group_col: Optional[str] = group_col
        self.other_cols_to_exclude: List[str] = other_cols_to_exclude if other_cols_to_exclude else []

        # Xác định các cột cần loại trừ tổng cộng
        self._cols_to_exclude_always: List[str] = self.target_cols + self.other_cols_to_exclude
        if self.group_col and self.group_col in self.data.columns:
            self._cols_to_exclude_always.append(self.group_col)
        # Loại bỏ các giá trị trùng lặp
        self._cols_to_exclude_always = list(set(self._cols_to_exclude_always))

        self.selected_features: Optional[List[str]] = None
        self.selection_method_used: Optional[str] = None
        self.selection_config_used: Optional[Dict[str, Any]] = None

        print(f"FeatureSelector initialized. Columns always excluded: {self._cols_to_exclude_always}")


    def _select_manual(self, feature_list: List[str]) -> List[str]:
        print(f"  Applying manual selection with list: {feature_list}")
        # Kiểm tra xem các feature trong list có tồn tại trong dữ liệu không
        available_cols = [col for col in self.data.columns if col not in self._cols_to_exclude_always]
        valid_features = []
        invalid_features = []
        for feature in feature_list:
            if feature in available_cols:
                valid_features.append(feature)
            elif feature in self._cols_to_exclude_always:
                 print(f"Warning: Feature '{feature}' is in the always excluded list, removing from manual selection.")
            else:
                invalid_features.append(feature)

        if invalid_features:
            print(f"Warning: The following manually specified features were not found or already excluded: {invalid_features}")
        if not valid_features:
             print("Error: No valid features remained after manual selection.")
             return []
        return valid_features

    def _select_by_correlation(self, threshold: float, target_col: str, analyzer: Optional['FeatureAnalyzer'] = None) -> List[str]:
        print(f"Applying correlation threshold selection (threshold={threshold}, target='{target_col}')...")
        if target_col not in self.target_cols:
            print(f"Warning: target_col '{target_col}' for correlation was not in the initial target_cols list.")
        if analyzer is None:
            print("Analyzer not provided, creating a temporary one...")
            analyzer = FeatureAnalyzer(self.data)
            analyzer.calculate_correlation() # Tính với cài đặt mặc định

        corr_series = analyzer.get_correlation_with_target(target_col)

        if corr_series is None:
            print(f"    Error: Could not get correlation with target '{target_col}'. Cannot select by correlation.")
            return []

        # Lấy giá trị tuyệt đối và lọc theo ngưỡng
        abs_corr = corr_series.abs()
        selected_by_corr = abs_corr[abs_corr >= threshold].index.tolist()

        # Loại bỏ các cột luôn bị loại trừ khỏi danh sách đã chọn
        final_selection = [f for f in selected_by_corr if f not in self._cols_to_exclude_always]

        if not final_selection:
             print(f"    Warning: No features met the correlation threshold >={threshold} with '{target_col}'.")

        return final_selection

    def _select_all_numeric(self) -> List[str]:
        print("  Applying selection of all numeric features...")
        numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        final_selection = [f for f in numeric_cols if f not in self._cols_to_exclude_always]
        if not final_selection:
            print("    Warning: No numeric features found after exclusions.")
        return final_selection


    def select(self,
               method: str,
               config: Optional[Dict[str, Any]] = None,
               feature_analyzer: Optional['FeatureAnalyzer'] = None) -> 'FeatureSelector':
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported selection method '{method}'. Supported methods are: {self.SUPPORTED_METHODS}")

        self.selection_method_used = method
        self.selection_config_used = config if config else {}
        selected = []
        print(f"\n--- Selecting features using method: {method} ---")

        if method == 'manual':
            if not config or 'feature_list' not in config or not isinstance(config['feature_list'], list):
                raise ValueError("Method 'manual' requires a 'feature_list' (list of strings) in config.")
            selected = self._select_manual(config['feature_list'])
        elif method == 'correlation_threshold':
            if not config or 'threshold' not in config or not isinstance(config['threshold'], (int, float)):
                 raise ValueError("Method 'correlation_threshold' requires a numeric 'threshold' in config.")
            if 'target_col' not in config or not isinstance(config['target_col'], str):
                 raise ValueError("Method 'correlation_threshold' requires a 'target_col' (string) in config.")
            # Truyền FeatureAnalyzer vào nếu có
            selected = self._select_by_correlation(config['threshold'], config['target_col'], analyzer=feature_analyzer)
        elif method == 'all_numeric':
            selected = self._select_all_numeric()

        self.selected_features = selected
        print(f"--- Feature selection finished. Selected {len(self.selected_features)} features ---")
        if self.selected_features:
             print(f"Selected Features: {self.selected_features}")
        else:
             print("Warning: No features were selected based on the criteria.")

        return self


    def get_selected_features(self) -> Optional[List[str]]:
        if self.selected_features is None:
            print("Warning: Features have not been selected yet. Call select() first.")
        return self.selected_features

    def get_data_with_selected_features(self, include_targets=False, include_group=False) -> Optional[pd.DataFrame]:
        selected = self.get_selected_features()
        if selected is None:
            return None

        cols_to_get = selected[:] # Tạo bản sao
        if include_targets:
            cols_to_get.extend([col for col in self.target_cols if col in self.data.columns])
        if include_group and self.group_col and self.group_col in self.data.columns:
            cols_to_get.append(self.group_col)

        # Loại bỏ trùng lặp và giữ thứ tự tương đối
        final_cols = list(pd.Series(cols_to_get).unique())

        return self.data[final_cols].copy()
