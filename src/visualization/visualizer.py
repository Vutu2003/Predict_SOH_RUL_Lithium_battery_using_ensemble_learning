# Class ResultVisualizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Any, Union, Tuple
from sklearn.metrics import mean_squared_error
# --- Giả sử các lớp khác đã được định nghĩa ---
# (Không cần import trực tiếp nhưng cần dữ liệu đầu ra từ chúng)

# --- Lớp 9 (hoặc lớp cuối cùng tùy cách đếm): ResultVisualizer ---

class ResultVisualizer:
    """
    Cung cấp các phương thức để trực quan hóa kết quả đánh giá mô hình
    và dự đoán SOH/RUL.
    """

    def __init__(self, style: str = 'seaborn-v0_8-whitegrid', palette: str = 'viridis'):
        """
        Khởi tạo ResultVisualizer.

        Args:
            style (str): Style của matplotlib/seaborn để sử dụng.
            palette (str): Bảng màu mặc định cho một số đồ thị.
        """
        self.style = style
        self.palette = palette
        print(f"ResultVisualizer initialized with style='{style}', palette='{palette}'.")

    def plot_actual_vs_predicted_scatter(self,
                                         results_df: pd.DataFrame,
                                         target_col: str, # 'SOH' hoặc 'RUL'
                                         title: Optional[str] = None,
                                         figsize: Tuple[int, int] = (7, 7),
                                         color_by: Optional[str] = None, # Tên cột để tô màu (vd: 'BatteryID', 'Fold')
                                         cmap: str = 'viridis',
                                         ax=None) -> None:
        """
        Vẽ đồ thị scatter plot so sánh giá trị thực tế và dự đoán.

        Args:
            results_df (pd.DataFrame): DataFrame chứa cột 'Actual' và 'Predicted'.
                                       Nên chứa cả cột target_col gốc nếu title tự động.
            target_col (str): Tên của biến mục tiêu ('SOH' hoặc 'RUL').
            title (Optional[str]): Tiêu đề đồ thị. Nếu None, tự tạo tiêu đề.
            figsize (Tuple[int, int]): Kích thước hình vẽ (nếu tạo figure mới).
            color_by (Optional[str]): Tên cột dùng để tô màu các điểm.
            cmap (str): Colormap nếu tô màu theo nhóm.
            ax (Optional[matplotlib.axes.Axes]): Axes để vẽ lên (nếu có).
        """
        if not isinstance(results_df, pd.DataFrame) or 'Actual' not in results_df or 'Predicted' not in results_df:
             print("Error plotting scatter: Input must be a DataFrame with 'Actual' and 'Predicted' columns.")
             return

        plt.style.use(self.style)
        create_figure = ax is None
        if create_figure:
            fig, ax = plt.subplots(figsize=figsize)

        actual_vals = results_df['Actual']
        pred_vals = results_df['Predicted']

        # Xác định màu sắc
        scatter_c = None
        legend_handles = None
        legend_labels = None
        if color_by and color_by in results_df.columns:
            colors_numeric, levels = pd.factorize(results_df[color_by])
            scatter_c = colors_numeric
            # Tạo handles cho legend
            scatter_proxy = plt.scatter([], [], c=[], cmap=cmap) # Tạo scatter ảo để lấy legend handles
            legend_handles, _ = scatter_proxy.legend_elements(prop="colors", num=len(levels), alpha=0.7)
            legend_labels = levels
            print(f"  Plotting points colored by '{color_by}'.")
        else:
             cmap = None # Không dùng cmap nếu không tô màu theo nhóm

        # Vẽ scatter plot
        scatter_plot = ax.scatter(actual_vals, pred_vals, alpha=0.7, edgecolors='k', s=40, c=scatter_c, cmap=cmap)

        # Vẽ đường y=x
        min_val = min(actual_vals.min(), pred_vals.min())
        max_val = max(actual_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction (y=x)')

        # Đặt tiêu đề và nhãn
        unit = "(%)" if target_col == 'SOH' else "(cycles)" if target_col == 'RUL' else ""
        ax.set_xlabel(f'Actual {target_col} {unit}')
        ax.set_ylabel(f'Predicted {target_col} {unit}')
        if title is None:
             # Tính RMSE nếu có thể để thêm vào title
             try:
                 rmse = np.sqrt(mean_squared_error(actual_vals, pred_vals))
                 title = f'{target_col}: Actual vs. Predicted\n(RMSE={rmse:.3f} {unit.strip()})'
             except:
                 title = f'{target_col}: Actual vs. Predicted'
        ax.set_title(title)

        # Thêm legend
        handles, labels = ax.get_legend_handles_labels()
        if legend_handles is not None: # Thêm legend cho màu sắc
             ax.legend(handles + legend_handles, labels + list(legend_labels), title=color_by, loc='best')
        else:
             ax.legend(loc='best')

        ax.grid(True)

        if create_figure:
            plt.tight_layout()
            plt.show()


    def plot_predictions_over_time(self,
                                   results_df: pd.DataFrame,
                                   time_col: str, # Tên cột thời gian ('cycle' hoặc index name)
                                   target_col: str, # 'SOH' hoặc 'RUL'
                                   group_col: Optional[str] = None, # Cột để vẽ riêng từng nhóm (vd: 'BatteryID')
                                   title: Optional[str] = None,
                                   figsize: Tuple[int, int] = (12, 6),
                                   ax=None) -> None:
        """
        Vẽ đồ thị đường so sánh giá trị thực tế và dự đoán theo thời gian/chu kỳ.

        Args:
            results_df (pd.DataFrame): DataFrame chứa 'Actual', 'Predicted', cột time_col,
                                       và tùy chọn cột group_col.
            time_col (str): Tên cột chứa giá trị trục X (ví dụ: 'cycle').
            target_col (str): Tên của biến mục tiêu ('SOH' hoặc 'RUL').
            group_col (Optional[str]): Nếu được cung cấp, sẽ vẽ các đường riêng biệt
                                       cho từng nhóm trong cột này.
            title (Optional[str]): Tiêu đề đồ thị. Nếu None, tự tạo tiêu đề.
            figsize (Tuple[int, int]): Kích thước hình vẽ (nếu tạo figure mới).
            ax (Optional[matplotlib.axes.Axes]): Axes để vẽ lên (nếu có).
        """
        required_cols = ['Actual', 'Predicted', time_col]
        if group_col: required_cols.append(group_col)
        if not isinstance(results_df, pd.DataFrame) or not all(c in results_df.columns for c in required_cols):
             print(f"Error plotting over time: DataFrame missing required columns ({required_cols}).")
             return

        plt.style.use(self.style)
        create_figure = ax is None
        if create_figure:
            fig, ax = plt.subplots(figsize=figsize)

        unit = "(%)" if target_col == 'SOH' else "(cycles)" if target_col == 'RUL' else ""
        plot_title = title if title is not None else f'{target_col} Prediction vs Actual over {time_col}'

        if group_col:
            # Vẽ riêng cho từng nhóm
            unique_groups = results_df[group_col].unique()
            colors = plt.cm.get_cmap(self.palette, len(unique_groups)) # Lấy màu từ palette
            print(f"  Plotting lines for each group in '{group_col}'...")
            for i, group_id in enumerate(unique_groups):
                subset = results_df[results_df[group_col] == group_id].sort_values(by=time_col)
                if not subset.empty:
                    ax.plot(subset[time_col], subset['Actual'], marker='.', linestyle='-', linewidth=1, color=colors(i), label=f'{group_id} Actual')
                    ax.plot(subset[time_col], subset['Predicted'], marker='x', linestyle='--', linewidth=1, color=colors(i), label=f'{group_id} Predicted')
            # Giới hạn số lượng legend nếu quá nhiều
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 10:
                 ax.legend(title=group_col, ncol=int(np.ceil(len(unique_groups)/5)), fontsize='small') # Chia cột legend
            else:
                 ax.legend(title=group_col)
        else:
            # Vẽ tổng thể
            df_sorted = results_df.sort_values(by=time_col)
            ax.plot(df_sorted[time_col], df_sorted['Actual'], label='Actual', marker='.', linestyle='-', linewidth=1.5)
            ax.plot(df_sorted[time_col], df_sorted['Predicted'], label='Predicted', marker='x', linestyle='--', linewidth=1)
            ax.legend()

        ax.set_xlabel(time_col.replace('_', ' ').title())
        ax.set_ylabel(f'{target_col} {unit}')
        ax.set_title(plot_title)
        ax.grid(True)

        if create_figure:
            plt.tight_layout()
            plt.show()

    def plot_feature_importance(self,
                                importance_df: pd.DataFrame,
                                title: Optional[str] = None,
                                figsize: Tuple[int, int] = (10, 8),
                                palette: Optional[str] = None, # Dùng palette của instance nếu None
                                ax=None) -> None:
        """
        Vẽ đồ thị cột thể hiện Feature Importance.

        Args:
            importance_df (pd.DataFrame): DataFrame có cột 'Feature' và 'Importance',
                                          đã được sắp xếp giảm dần.
            title (Optional[str]): Tiêu đề đồ thị.
            figsize (Tuple[int, int]): Kích thước hình vẽ.
            palette (Optional[str]): Bảng màu.
            ax (Optional[matplotlib.axes.Axes]): Axes để vẽ lên.
        """
        if not isinstance(importance_df, pd.DataFrame) or 'Feature' not in importance_df or 'Importance' not in importance_df:
             print("Error plotting importance: Input must be a DataFrame with 'Feature' and 'Importance' columns.")
             return

        plt.style.use(self.style)
        create_figure = ax is None
        if create_figure:
            # Điều chỉnh chiều cao dựa trên số lượng feature
            height = max(6, len(importance_df) * 0.4)
            fig, ax = plt.subplots(figsize=(figsize[0], height))

        plot_palette = palette if palette is not None else self.palette
        plot_title = title if title is not None else 'Feature Importance'

        sns.barplot(x='Importance', y='Feature', data=importance_df,
                    palette=plot_palette, orient='h', ax=ax)
        ax.set_title(plot_title)
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Features')

        if create_figure:
            plt.tight_layout()
            plt.show()

    def plot_cv_metric_distribution(self,
                                    cv_results: Dict[str, List[float]], # Dict từ CrossValidator.cv_results_
                                    figsize: Tuple[int, int] = (12, 5),
                                    **kwargs) -> None:
        """
        Vẽ biểu đồ hộp thể hiện phân phối của các metric qua các fold CV.

        Args:
            cv_results (Dict[str, List[float]]): Dictionary chứa kết quả metrics từ CV,
                                                 ví dụ: {'mae': [..], 'rmse': [..], 'r2': [..]}.
                                                 Giá trị lỗi (mae, rmse) nên là dương.
            figsize (Tuple[int, int]): Kích thước hình vẽ.
            **kwargs: Các tham số khác cho sns.boxplot.
        """
        if not isinstance(cv_results, dict) or not cv_results:
            print("Error plotting CV metrics: Input cv_results must be a non-empty dictionary.")
            return

        plt.style.use(self.style)
        num_metrics = len(cv_results)
        fig, axes = plt.subplots(1, num_metrics, figsize=figsize, squeeze=False) # Đảm bảo axes là 2D
        axes = axes.flatten() # Làm phẳng

        print("\nPlotting CV metric distributions...")
        i = 0
        for metric, scores in cv_results.items():
            if i >= len(axes): break # Đề phòng nhiều metric hơn subplot
            ax = axes[i]
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                sns.boxplot(y=valid_scores, ax=ax, palette=self.palette, **kwargs)
                sns.stripplot(y=valid_scores, ax=ax, color='black', size=5, alpha=0.7)
                unit = "(cycles)" if metric in ['mae', 'rmse'] and 'RUL' in plt.gca().get_title() else \
                       "(%)" if metric in ['mae', 'rmse'] and 'SOH' in plt.gca().get_title() else ""
                ax.set_title(f'{metric.upper()} Distribution (CV)')
                ax.set_ylabel(f'{metric.upper()} {unit}')
            else:
                 ax.set_title(f'{metric.upper()} (No valid data)')
            ax.grid(True)
            i += 1

        # Ẩn các axes thừa
        for j in range(i, len(axes)):
             axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()
