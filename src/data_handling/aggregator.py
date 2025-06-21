# Clas DataAggregator
import pandas as pd
import numpy as np
import os
from tqdm import tqdm # Để hiển thị tiến trình tổng thể
from typing import List, Dict, Optional, Any
from reader import BatteryDataReader
from feature_engineer import BatteryFeatureEngineer


class DataAggregator:
    def __init__(self,
                 battery_ids: List[str],
                 data_dir: str,
                 fe_config: Optional[Dict[str, Any]] = None):
        if not isinstance(battery_ids, list) or not all(isinstance(bid, str) for bid in battery_ids):
            raise ValueError("battery_ids phải là một danh sách các chuỗi.")
        if not battery_ids:
             raise ValueError("battery_ids không được rỗng.")

        self.battery_ids: List[str] = list(set(battery_ids)) # Đảm bảo ID là duy nhất
        self.data_dir: str = data_dir
        self.fe_config: Dict[str, Any] = fe_config if fe_config is not None else {}
        self.combined_data: Optional[pd.DataFrame] = None
        self._processed_individual_data: Dict[str, Optional[pd.DataFrame]] = {} # Lưu trữ kết quả từng pin (tùy chọn)

    def aggregate(self, force_rerun: bool = False) -> 'DataAggregator':
        if self.combined_data is not None and not force_rerun:
            print("Combined data already exists. Skipping aggregation. Use force_rerun=True to re-aggregate.")
            return self

        print(f"\n--- Starting Data Aggregation for Batteries: {self.battery_ids} ---")
        self._processed_individual_data = {} # Xóa dữ liệu cũ nếu chạy lại
        all_processed_dfs = []

        for battery_id in tqdm(self.battery_ids, desc="Aggregating Batteries"):
            print(f"\n--- Processing: {battery_id} ---")
            # 1. Đọc dữ liệu thô
            reader = BatteryDataReader(battery_id, self.data_dir)
            if not reader.load_data():
                print(f"  Skipping {battery_id} due to data loading failure.")
                self._processed_individual_data[battery_id] = None
                continue # Chuyển sang pin tiếp theo

            cap_df, chg_df, dis_df = reader.get_raw_data()

            # Kiểm tra xem có đủ dữ liệu không (ít nhất capacity phải có)
            if cap_df is None or chg_df is None or dis_df is None:
                 print(f"  Skipping {battery_id} due to missing essential raw data (Capacity, Charge, or Discharge).")
                 self._processed_individual_data[battery_id] = None
                 continue

            # 2. Thực hiện Feature Engineering
            feature_engineer = BatteryFeatureEngineer(
                battery_id=battery_id,
                capacity_df=cap_df,
                charge_df=chg_df,
                discharge_df=dis_df,
                config=self.fe_config
            )
            feature_engineer.process() # Chạy quy trình FE
            processed_df = feature_engineer.get_processed_data()

            # 3. Lưu trữ và thêm vào danh sách kết hợp
            if processed_df is not None and not processed_df.empty:
                print(f"  Successfully processed {battery_id}. Shape: {processed_df.shape}")
                all_processed_dfs.append(processed_df)
                self._processed_individual_data[battery_id] = processed_df
            else:
                print(f"  Feature engineering failed or produced empty DataFrame for {battery_id}.")
                self._processed_individual_data[battery_id] = None

        # 4. Kết hợp tất cả DataFrame đã xử lý
        if not all_processed_dfs:
            print("\nError: No battery data was successfully processed and aggregated.")
            self.combined_data = None
            return self

        print("\n--- Concatenating data from all processed batteries ---")
        self.combined_data = pd.concat(all_processed_dfs, ignore_index=True)
        print(f"Combined DataFrame shape: {self.combined_data.shape}")
        print(f"Unique batteries in combined data: {self.combined_data['battery_id'].unique()}")

        # 5. Xử lý NaN cuối cùng (tùy chọn, nhưng nên có)
        print("\n--- Performing final NaN check and fill on combined data ---")
        initial_nan_count = self.combined_data.isnull().sum().sum()
        if initial_nan_count > 0:
            print(f"Found {initial_nan_count} NaN values before final fill.")
            self.combined_data = self.combined_data.fillna(method='bfill').fillna(method='ffill')
            final_nan_count = self.combined_data.isnull().sum().sum()
            print(f"NaN count after final fill: {final_nan_count}")
            if final_nan_count > 0:
                print("Warning: Some NaNs might still remain. Consider further imputation if needed.")
        else:
             print("No NaNs found in the combined data after individual processing.")

        print("--- Data Aggregation Finished ---")
        return self

    def get_combined_data(self) -> Optional[pd.DataFrame]:
        if self.combined_data is None:
            print("Warning: Combined data not available. Call aggregate() first.")
        return self.combined_data

    def get_processed_data_for_battery(self, battery_id: str) -> Optional[pd.DataFrame]:
        if battery_id not in self._processed_individual_data:
            print(f"Warning: Data for battery {battery_id} was not processed or processing failed.")
            return None
        return self._processed_individual_data[battery_id]
