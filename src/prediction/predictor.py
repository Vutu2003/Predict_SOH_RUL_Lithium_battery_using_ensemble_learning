# Class Predictor
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
# Import các lớp cần thiết khác (giả sử đã định nghĩa)
# from your_module import BatteryFeatureEngineer, ModelTrainer # Hoặc định nghĩa lại ở đây nếu cần

# --- Giả sử các lớp BatteryFeatureEngineer và ModelTrainer đã tồn tại ---
# class BatteryFeatureEngineer: ...
# class ModelTrainer: ...
# (Lưu ý: Predictor không trực tiếp cần ModelTrainer, mà cần mô hình đã huấn luyện từ nó)
from sklearn.base import BaseEstimator # Import BaseEstimator để kiểm tra kiểu model
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data_handling')))
from feature_engineer import BatteryFeatureEngineer

# --- Lớp 10: Predictor ---

class Predictor:
    """
    Sử dụng các mô hình SOH và RUL đã huấn luyện để dự đoán trên
    dữ liệu thô của một chu kỳ mới từ một pin.
    """

    def __init__(self,
                 soh_model: BaseEstimator,
                 rul_model: BaseEstimator,
                 fe_config: Optional[Dict[str, Any]], # Config đã dùng để train FE
                 soh_feature_names: List[str],      # Features cần cho model SOH
                 rul_feature_names: List[str]       # Features cần cho model RUL (bao gồm 'SOH')
                ):
        """
        Khởi tạo Predictor.

        Args:
            soh_model (BaseEstimator): Đối tượng mô hình SOH đã huấn luyện
                                        (ví dụ: từ final_soh_trainer.get_model()).
            rul_model (BaseEstimator): Đối tượng mô hình RUL đã huấn luyện
                                        (ví dụ: từ final_rul_trainer.get_model()).
            fe_config (Optional[Dict[str, Any]]): Dictionary cấu hình đã sử dụng
                                                   khi chạy Feature Engineering
                                                   để huấn luyện các mô hình này.
            soh_feature_names (List[str]): Danh sách chính xác các tên feature
                                           mà mô hình SOH yêu cầu làm đầu vào.
            rul_feature_names (List[str]): Danh sách chính xác các tên feature
                                           mà mô hình RUL yêu cầu làm đầu vào
                                           (phải bao gồm cột 'SOH').
        """
        if not isinstance(soh_model, BaseEstimator) or not hasattr(soh_model, 'predict'):
             raise TypeError("soh_model phải là một đối tượng mô hình scikit-learn đã huấn luyện.")
        if not isinstance(rul_model, BaseEstimator) or not hasattr(rul_model, 'predict'):
             raise TypeError("rul_model phải là một đối tượng mô hình scikit-learn đã huấn luyện.")
        if not isinstance(soh_feature_names, list) or not soh_feature_names:
            raise ValueError("soh_feature_names phải là một list các tên feature không rỗng.")
        if not isinstance(rul_feature_names, list) or not rul_feature_names:
            raise ValueError("rul_feature_names phải là một list các tên feature không rỗng.")
        if 'SOH' not in rul_feature_names:
             print("Warning: 'SOH' is expected to be in rul_feature_names for realistic RUL prediction.")


        self.soh_model = soh_model
        self.rul_model = rul_model
        # Lưu config FE để đảm bảo tính nhất quán khi xử lý dữ liệu mới
        self.fe_config = fe_config if fe_config is not None else BatteryFeatureEngineer.DEFAULT_CONFIG # Dùng default nếu không có
        self.soh_feature_names = soh_feature_names
        self.rul_feature_names = rul_feature_names

        print("Predictor initialized.")
        print(f"  SOH model type: {type(self.soh_model).__name__}")
        print(f"  RUL model type: {type(self.rul_model).__name__}")
        print(f"  SOH features required ({len(self.soh_feature_names)}): {self.soh_feature_names}")
        print(f"  RUL features required ({len(self.rul_feature_names)}): {self.rul_feature_names}")


    def _preprocess_new_data(self,
                             battery_id: str, # Cần ID để khởi tạo FE
                             capacity_df: Optional[pd.DataFrame],
                             charge_df: Optional[pd.DataFrame],
                             discharge_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        (Private) Áp dụng quy trình Feature Engineering cho dữ liệu mới.

        Sử dụng cùng logic FE (thông qua BatteryFeatureEngineer và fe_config)
        như khi huấn luyện mô hình.

        Returns:
            Optional[pd.DataFrame]: DataFrame chứa các features đã tính toán cho
                                   dữ liệu đầu vào, hoặc None nếu lỗi.
                                   Thường chỉ chứa 1 hàng nếu input là 1 chu kỳ.
        """
        print(f"\n  Preprocessing new data for battery: {battery_id}...")
        # Kiểm tra dữ liệu đầu vào cơ bản
        if capacity_df is None or charge_df is None or discharge_df is None:
            print("  Error: Missing one or more essential raw dataframes (capacity, charge, discharge) for preprocessing.")
            return None
        if capacity_df.empty or charge_df.empty or discharge_df.empty:
            print("  Warning: One or more input dataframes are empty.")
            # Có thể vẫn tiếp tục nếu FE xử lý được, nhưng nên cẩn thận

        try:
            # Tạo instance FE tạm thời để xử lý dữ liệu mới
            temp_feature_engineer = BatteryFeatureEngineer(
                battery_id=battery_id, # ID này có thể là 'UNKNOWN' nếu không biết
                capacity_df=capacity_df,
                charge_df=charge_df,
                discharge_df=discharge_df,
                config=self.fe_config
            )
            temp_feature_engineer.process()
            processed_new_data = temp_feature_engineer.get_processed_data()

            if processed_new_data is None or processed_new_data.empty:
                 print(f"  Error: Feature Engineering failed or produced empty result for {battery_id}.")
                 return None

            print("  Preprocessing finished successfully.")
            # Trả về toàn bộ kết quả xử lý (có thể có nhiều hơn 1 hàng nếu input là nhiều chu kỳ)
            return processed_new_data

        except Exception as e:
            print(f"  Error during preprocessing new data: {e}")
            return None


    def predict(self,
                battery_id: str,
                capacity_df: pd.DataFrame,
                charge_df: pd.DataFrame,
                discharge_df: pd.DataFrame,
                cycle_index: Optional[int] = -1 # Chỉ số của chu kỳ cần dự đoán (-1 là cuối cùng)
               ) -> Tuple[Optional[float], Optional[float]]:
        """
        Dự đoán SOH và RUL cho một chu kỳ dữ liệu cụ thể.

        Args:
            battery_id (str): ID (hoặc tên định danh) cho dữ liệu này.
            capacity_df (pd.DataFrame): Dữ liệu capacity thô (có thể chỉ chứa chu kỳ hiện tại).
            charge_df (pd.DataFrame): Dữ liệu charge thô của chu kỳ hiện tại.
            discharge_df (pd.DataFrame): Dữ liệu discharge thô của chu kỳ hiện tại.
            cycle_index (Optional[int]): Chỉ số hàng (-1: cuối cùng, 0: đầu tiên, ...)
                                         trong DataFrame đã xử lý để lấy feature.
                                         Mặc định là -1 (chu kỳ mới nhất).

        Returns:
            Tuple[Optional[float], Optional[float]]: Tuple chứa (soh_predicted, rul_predicted).
                                                   Giá trị có thể là None nếu dự đoán thất bại.
        """
        print(f"\n--- Predicting SOH and RUL for {battery_id} (cycle index: {cycle_index}) ---")

        # 1. Tiền xử lý dữ liệu mới
        processed_data = self._preprocess_new_data(battery_id, capacity_df, charge_df, discharge_df)

        if processed_data is None:
            print("Prediction failed due to preprocessing error.")
            return None, None

        # 2. Lấy đúng hàng dữ liệu cần dự đoán
        try:
            # Sử dụng iloc để lấy theo chỉ số vị trí
            if cycle_index is not None and abs(cycle_index) < len(processed_data):
                 data_to_predict = processed_data.iloc[[cycle_index]] # Giữ dạng DataFrame (2D)
            elif cycle_index is None and len(processed_data) == 1:
                 # Nếu chỉ có 1 hàng và index là None, lấy hàng đó
                  data_to_predict = processed_data.iloc[[0]]
            else:
                 print(f"  Error: Invalid cycle_index ({cycle_index}) for processed data with length {len(processed_data)}.")
                 return None, None
        except IndexError:
             print(f"  Error: Index {cycle_index} out of bounds for processed data.")
             return None, None

        print(f"  Data selected for prediction (shape {data_to_predict.shape}):\n{data_to_predict}")


        soh_pred: Optional[float] = None
        rul_pred: Optional[float] = None

        # 3. Dự đoán SOH
        print("\n  Predicting SOH...")
        try:
            # Chuẩn bị input X cho SOH (chọn đúng cột và đúng thứ tự)
            X_soh_predict = data_to_predict[self.soh_feature_names]
            soh_pred_array = self.soh_model.predict(X_soh_predict)
            soh_pred = float(soh_pred_array[0]) # Lấy giá trị đầu tiên và chuyển thành float
            print(f"  Predicted SOH: {soh_pred:.2f} %")
        except KeyError as e:
             print(f"  Error preparing SOH features for prediction: Missing column {e}")
             # Không thể tiếp tục dự đoán RUL nếu SOH lỗi
             return None, None
        except Exception as e:
            print(f"  Error during SOH prediction: {e}")
            # Không thể tiếp tục dự đoán RUL nếu SOH lỗi
            return None, None


        # 4. Dự đoán RUL (sử dụng SOH vừa dự đoán)
        print("\n  Predicting RUL...")
        if soh_pred is None:
             print("  Cannot predict RUL because SOH prediction failed.")
             return None, None

        try:
            # Chuẩn bị input X cho RUL
            X_rul_predict_base = data_to_predict[self.rul_feature_names].copy() # Lấy các cột cần thiết
            # Đảm bảo cột 'SOH' tồn tại trước khi gán
            if 'SOH' not in X_rul_predict_base.columns:
                 print("   Creating 'SOH' column for RUL prediction.")
                 # Hoặc raise lỗi nếu SOH bắt buộc phải có từ FE
                 # raise ValueError("'SOH' column missing in preprocessed data for RUL features.")
            # Gán giá trị SOH dự đoán vào
            X_rul_predict_base['SOH'] = soh_pred
            # Đảm bảo đúng thứ tự cột như khi huấn luyện RUL model
            X_rul_predict = X_rul_predict_base[self.rul_feature_names]

            rul_pred_array = self.rul_model.predict(X_rul_predict)
            rul_pred = float(rul_pred_array[0]) # Lấy giá trị đầu tiên
            # Có thể làm tròn hoặc giới hạn RUL nếu cần (ví dụ: không âm)
            rul_pred = max(0.0, round(rul_pred)) # Làm tròn thành số nguyên không âm
            print(f"  Predicted RUL: {rul_pred:.0f} cycles")
        except KeyError as e:
             print(f"  Error preparing RUL features for prediction: Missing column {e}")
             rul_pred = None
        except Exception as e:
            print(f"  Error during RUL prediction: {e}")
            rul_pred = None

        print("\n--- Prediction Finished ---")
        return soh_pred, rul_pred
