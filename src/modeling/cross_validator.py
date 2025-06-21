# Class CrossValidator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.base import clone, BaseEstimator
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Union
from evaluator import ModelEvaluator
from trainer import ModelTrainer

# --- Giả sử các lớp ModelEvaluator và ModelTrainer đã được định nghĩa ---
# class ModelEvaluator: ...
# class ModelTrainer: ...
# --- Giả sử df_selected_all và final_feature_list đã tồn tại ---

# --- Lớp 8: CrossValidator (Cập nhật để vẽ đồ thị từng fold) ---

class CrossValidator:
    SUPPORTED_CV_STRATEGIES = ['LOBO', 'GroupKFold']

    def __init__(self,
                 data: pd.DataFrame,
                 feature_cols: List[str],
                 target_col: str,
                 group_col: str,
                 model_trainer: 'ModelTrainer',
                 evaluator: 'ModelEvaluator',
                 cv_strategy: str = 'LOBO',
                 n_splits_gkf: Optional[int] = None,
                 time_col: str = 'cycle' # Thêm tham số cột thời gian/chu kỳ
                ):
        # ... (Phần __init__ còn lại giữ nguyên như trước) ...
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Input 'data' phải là một DataFrame không rỗng.")
        if cv_strategy not in self.SUPPORTED_CV_STRATEGIES:
            raise ValueError(f"Unsupported cv_strategy '{cv_strategy}'. Supported: {self.SUPPORTED_CV_STRATEGIES}")
        if not isinstance(model_trainer, ModelTrainer):
            raise TypeError("model_trainer phải là một instance của ModelTrainer.")
        if not isinstance(evaluator, ModelEvaluator):
             raise TypeError("evaluator phải là một instance của ModelEvaluator.")

        required_cols = feature_cols + [target_col, group_col, time_col] # Thêm time_col vào kiểm tra
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            # Nếu time_col không bắt buộc, có thể chỉ cảnh báo
            if time_col in missing_cols and len(missing_cols) == 1:
                 print(f"Warning: Time column '{time_col}' not found. Plots over time will use index.")
                 self.time_col = None # Đặt thành None nếu không tìm thấy
                 required_cols.remove(time_col) # Bỏ qua kiểm tra time_col
                 # Kiểm tra lại các cột còn lại
                 missing_cols = [col for col in required_cols if col not in data.columns]
                 if missing_cols:
                     raise ValueError(f"Các cột bắt buộc sau bị thiếu: {missing_cols}")
            else:
                raise ValueError(f"Các cột bắt buộc sau bị thiếu: {missing_cols}")
        else:
             self.time_col = time_col # Lưu lại tên cột thời gian

        self.data = data.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.group_col = group_col
        self.model_trainer = model_trainer
        self.evaluator = evaluator
        self.cv_strategy_name = cv_strategy
        self.n_splits_gkf = n_splits_gkf
        self._cv_splitter = self._get_cv_splitter()
        self.n_splits = self._cv_splitter.get_n_splits(self.data[self.feature_cols], self.data[self.target_col], self.data[self.group_col])
        self.cv_results_: Optional[Dict[str, List[float]]] = None
        self.cv_predictions_: Optional[pd.DataFrame] = None

        print(f"CrossValidator initialized for target '{self.target_col}' using {self.cv_strategy_name} ({self.n_splits} splits).")
        print(f"Model to evaluate: {self.model_trainer.model_type}")
        if self.time_col:
            print(f"Using '{self.time_col}' column for plotting over time.")

    def _get_cv_splitter(self):
        # ... (Giữ nguyên như trước) ...
        if self.cv_strategy_name == 'LOBO':
            return LeaveOneGroupOut()
        elif self.cv_strategy_name == 'GroupKFold':
            n_groups = self.data[self.group_col].nunique()
            splits = self.n_splits_gkf if self.n_splits_gkf is not None else n_groups
            if splits > n_groups: splits = n_groups
            elif splits <= 1: raise ValueError("n_splits_gkf must be >= 2.")
            return GroupKFold(n_splits=splits)
        else: raise ValueError(f"Invalid cv_strategy_name: {self.cv_strategy_name}")


    def run(self, plot_each_fold: bool = True) -> 'CrossValidator':
        print(f"\n--- Running {self.cv_strategy_name} CV for target '{self.target_col}' (Plot each fold: {plot_each_fold}) ---")

        X = self.data[self.feature_cols]
        y = self.data[self.target_col]
        groups = self.data[self.group_col]
        # Lấy cột thời gian nếu có, nếu không dùng index
        time_data = self.data[self.time_col] if self.time_col else pd.Series(self.data.index, index=self.data.index)


        predictions_list = []
        fold_metrics = {metric: [] for metric in self.evaluator.metrics_to_calculate}
        fold_count = 0

        # --- Chuẩn bị Figure cho đồ thị từng fold (NẾU plot_each_fold=True) ---
        fig_folds, axes_folds = (None, None)
        if plot_each_fold:
            num_cols_plot = 2
            num_rows_plot = int(np.ceil(self.n_splits / num_cols_plot))
            fig_folds, axes_folds = plt.subplots(num_rows_plot, num_cols_plot,
                                                 figsize=(7 * num_cols_plot, 5 * num_rows_plot),
                                                 squeeze=False)
            fig_folds.suptitle(f'{self.target_col} Prediction vs Actual per Fold ({self.cv_strategy_name} CV)', fontsize=16)
            axes_flat = axes_folds.flatten() # Mảng 1D các axes

        # --- Vòng lặp CV ---
        cv_iterator = self._cv_splitter.split(X, y, groups)
        for train_index, test_index in tqdm(cv_iterator, total=self.n_splits, desc=f"{self.cv_strategy_name} Folds"):
            current_fold_idx = fold_count # Index cho subplot (0-based)
            fold_count += 1
            test_groups = groups.iloc[test_index].unique()
            left_out_info = f"Group(s): {list(test_groups)}"

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            time_test = time_data.iloc[test_index] # Lấy dữ liệu thời gian/chu kỳ cho tập test

            # --- Huấn luyện ---
            try:
                 trainer_fold = clone(self.model_trainer)
            except TypeError:
                 print(f"Warning: Could not clone ModelTrainer in Fold {fold_count}. Reusing instance.")
                 trainer_fold = self.model_trainer
            try:
                trainer_fold.train(X_train, y_train)
                model_fold = trainer_fold.get_model()
                if model_fold is None: raise RuntimeError("Training failed.")
            except Exception as e:
                print(f"  ERROR during training in Fold {fold_count}: {e}. Skipping.")
                for metric in fold_metrics: fold_metrics[metric].append(np.nan)
                continue

            # --- Dự đoán ---
            y_pred_fold = trainer_fold.predict(X_test)
            if y_pred_fold is None:
                 print(f"  ERROR during prediction in Fold {fold_count}. Skipping.")
                 for metric in fold_metrics: fold_metrics[metric].append(np.nan)
                 continue

            # --- Đánh giá ---
            fold_result_metrics = self.evaluator.calculate_metrics(y_test, y_pred_fold)
            for metric in fold_metrics:
                fold_metrics[metric].append(fold_result_metrics.get(metric, np.nan))

            # --- Lưu dự đoán ---
            predictions_list.append(pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': y_pred_fold,
                'Group': groups.iloc[test_index].values,
                'Fold': fold_count,
                self.time_col if self.time_col else 'Index': time_test.values # Dùng tên cột time_col hoặc 'Index'
            }, index=X_test.index)) # Giữ index gốc của DataFrame

            # --- Vẽ đồ thị cho Fold hiện tại (NẾU plot_each_fold=True) ---
            if plot_each_fold and fig_folds is not None:
                 if current_fold_idx < len(axes_flat): # Đảm bảo không vượt quá số subplot
                    ax = axes_flat[current_fold_idx]
                    fold_results_df = pd.DataFrame({
                        'Actual': y_test,
                        'Predicted': y_pred_fold,
                        'Time': time_test # Trục X là time_col hoặc index
                    }).sort_values('Time')

                    fold_rmse = fold_result_metrics.get('rmse', np.nan)
                    fold_r2 = fold_result_metrics.get('r2', np.nan)
                    fold_mae = fold_result_metrics.get('mae',np.nan)
                    ax.plot(fold_results_df['Time'], fold_results_df['Actual'], label='Actual', marker='.', linestyle='-', linewidth=1)
                    ax.plot(fold_results_df['Time'], fold_results_df['Predicted'], label='Predicted', marker='x', linestyle='--', linewidth=1)
                    ax.set_title(f'Fold {fold_count}: Out {list(test_groups)}\nRMSE={fold_rmse:.3f}, R²={fold_r2:.3f}, MAE={fold_mae:.3f}')
                    ax.set_xlabel(self.time_col if self.time_col else 'Index')
                    ax.set_ylabel(self.target_col)
                    ax.legend(fontsize='small')
                    ax.grid(True)
                    # Điều chỉnh giới hạn Y nếu cần
                    min_y = min(fold_results_df['Actual'].min(), fold_results_df['Predicted'].min())
                    max_y = max(fold_results_df['Actual'].max(), fold_results_df['Predicted'].max())
                    padding = (max_y - min_y) * 0.05 # Thêm chút padding
                    ax.set_ylim(min_y - padding, max_y + padding)
                 else:
                      print(f"Warning: Not enough subplots created for fold {fold_count}.")


        print(f"{self.cv_strategy_name} Cross-validation loop finished.")

        # --- Hiển thị đồ thị các fold (NẾU plot_each_fold=True) ---
        if plot_each_fold and fig_folds is not None:
            # Ẩn các subplot trống
            for i in range(fold_count, len(axes_flat)):
                axes_flat[i].set_visible(False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # --- Lưu kết quả cuối cùng ---
        self.cv_results_ = fold_metrics
        self.cv_predictions_ = pd.concat(predictions_list).sort_index() if predictions_list else None

        return self, predictions_list

    def get_metrics_summary(self) -> Optional[pd.DataFrame]:
        # ... (Giữ nguyên như trước) ...
        if self.cv_results_ is None: return None
        summary = {}
        print("\n--- Cross-Validation Metrics Summary ---")
        for metric, scores in self.cv_results_.items():
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                 mean_score, std_score = np.mean(valid_scores), np.std(valid_scores)
                 summary[metric] = {'mean': mean_score, 'std': std_score}
                 unit = "(cycles)" if metric in ['mae', 'rmse'] and self.target_col == 'RUL' else \
                        "(%)" if metric in ['mae', 'rmse'] and self.target_col == 'SOH' else ""
                 print(f"  Mean {metric.upper()}: {mean_score:.4f} +/- {std_score:.4f} {unit}")
            else:
                 summary[metric] = {'mean': np.nan, 'std': np.nan}
                 print(f"  Mean {metric.upper()}: NaN")
        return pd.DataFrame.from_dict(summary, orient='index')

    def get_predictions(self) -> Optional[pd.DataFrame]:
        # ... (Giữ nguyên như trước) ...
        return self.cv_predictions_
