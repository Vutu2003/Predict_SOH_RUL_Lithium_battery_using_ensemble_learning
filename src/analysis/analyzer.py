# Class FeatureAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Any, List # Thêm List


class FeatureAnalyzer:
    def __init__(self, combined_data: pd.DataFrame):
        if not isinstance(combined_data, pd.DataFrame) or combined_data.empty:
            raise ValueError("Input 'combined_data' phải là một DataFrame không rỗng.")
        # Tạo bản sao để tránh thay đổi DataFrame gốc
        self.data: pd.DataFrame = combined_data.copy()
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self._numeric_columns: Optional[List[str]] = None # Lưu trữ các cột số đã dùng

        print(f"FeatureAnalyzer initialized with data shape: {self.data.shape}")

    def calculate_correlation(self,
                              method: str = 'pearson',
                              numeric_only: bool = True) -> 'FeatureAnalyzer':
        print(f"\nCalculating correlation matrix (method='{method}', numeric_only={numeric_only})...")
        data_to_correlate = self.data

        if numeric_only:
            # Chọn các cột số, loại trừ các cột không phải số như 'battery_id'
            self._numeric_columns = self.data.select_dtypes(include=np.number).columns.tolist()
            if not self._numeric_columns:
                 print("Warning: No numeric columns found to calculate correlation.")
                 self.correlation_matrix = None
                 return self
            data_to_correlate = self.data[self._numeric_columns]
            print(f"  Using {len(self._numeric_columns)} numeric columns for correlation.")
        else:
            # Cảnh báo nếu không chỉ dùng cột số vì .corr() có thể lỗi
            print("  Warning: Calculating correlation on potentially non-numeric columns.")
            self._numeric_columns = list(self.data.columns) # Giả sử tất cả đều dùng được

        try:
            self.correlation_matrix = data_to_correlate.corr(method=method)
            print("  Correlation matrix calculated successfully.")
        except Exception as e:
            print(f"  Error calculating correlation matrix: {e}")
            self.correlation_matrix = None

        return self # Cho phép chaining

    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        if self.correlation_matrix is None:
            print("Correlation matrix not calculated yet. Calculating with default settings...")
            self.calculate_correlation() # Gọi với default pearson, numeric_only=True

        return self.correlation_matrix

    def plot_correlation_heatmap(self,
                                 figsize: Tuple[int, int] = (15, 12), # Giảm kích thước mặc định một chút
                                 cmap: str = 'coolwarm',
                                 annot: bool = False,
                                 annot_kws: Optional[Dict[str, Any]] = None,
                                 fontsize: int = 8,
                                 **kwargs) -> None:
        corr_matrix = self.get_correlation_matrix()
        if corr_matrix is None:
            print("Error: Cannot plot heatmap. Correlation matrix is not available.")
            return

        print("\nPlotting correlation heatmap...")
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=figsize)

        # Điều chỉnh kích thước annotation nếu hiển thị
        default_annot_kws = {"size": fontsize - 2} # Làm chữ nhỏ hơn label một chút
        if annot_kws:
            default_annot_kws.update(annot_kws)

        sns.heatmap(corr_matrix,
                    annot=annot,
                    cmap=cmap,
                    linewidths=0.5,
                    linecolor='lightgrey',
                    fmt=".2f", # Định dạng số nếu annot=True
                    annot_kws=default_annot_kws if annot else None,
                    **kwargs)

        plt.title('Feature Correlation Matrix', fontsize=fontsize + 4)
        plt.xticks(rotation=45, ha='right', fontsize=fontsize)
        plt.yticks(rotation=0, fontsize=fontsize)
        plt.tight_layout()
        plt.show()

    def get_correlation_with_target(self, target_col: str) -> Optional[pd.Series]:
        corr_matrix = self.get_correlation_matrix()
        if corr_matrix is None:
            print(f"Error: Correlation matrix not available to get correlation with '{target_col}'.")
            return None
        if target_col not in corr_matrix.columns:
            print(f"Error: Target column '{target_col}' not found in the calculated correlation matrix (available: {list(corr_matrix.columns)}).")
            return None

        return corr_matrix[target_col].sort_values(ascending=False) # Trả về đã sắp xếp

    def plot_correlation_with_target(self,
                                     target_col: str,
                                     sort_by_abs: bool = True,
                                     figsize: Tuple[int, int] = (10, 8),
                                     palette: str = "coolwarm_r", # Đảo ngược coolwarm để dương là xanh, âm là đỏ
                                     **kwargs) -> None:
        corr_series = self.get_correlation_with_target(target_col)
        if corr_series is None:
            print(f"Error: Cannot plot correlation with '{target_col}'.")
            return

        print(f"\nPlotting correlation with target: '{target_col}'...")
        # Loại bỏ chính target ra khỏi series để vẽ
        corr_series_plot = corr_series.drop(target_col, errors='ignore')

        if sort_by_abs:
            # Sắp xếp lại theo giá trị tuyệt đối giảm dần
            corr_series_plot = corr_series_plot.abs().sort_values(ascending=False)
            # Lấy lại giá trị gốc (có dấu) theo thứ tự mới
            corr_series_plot = corr_series.loc[corr_series_plot.index]


        plt.figure(figsize=figsize)
        sns.barplot(x=corr_series_plot.values, y=corr_series_plot.index,
                    palette=palette, orient='h', **kwargs)
        plt.title(f'Feature Correlation with {target_col}')
        plt.xlabel('Pearson Correlation Coefficient')
        plt.ylabel('Features')
        plt.axvline(0, color='black', linewidth=0.6) # Đường zero
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()