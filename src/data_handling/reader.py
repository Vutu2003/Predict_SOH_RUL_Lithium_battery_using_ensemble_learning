
# Class BatteryDataReader
import pandas as pd
import os
from typing import Tuple, Optional # Đảm bảo import các kiểu dữ liệu cần thiết

class BatteryDataReader:
    def __init__(self, battery_id: str, data_dir: str):
        if not isinstance(battery_id, str) or not battery_id:
            raise ValueError("battery_id phải là một chuỗi không rỗng.")
        if not isinstance(data_dir, str):
             raise ValueError("data_dir phải là một chuỗi.")

        self.battery_id: str = battery_id
        self.data_dir: str = data_dir
        self.capacity_df: Optional[pd.DataFrame] = None
        self.charge_df: Optional[pd.DataFrame] = None
        self.discharge_df: Optional[pd.DataFrame] = None
        self._data_loaded: bool = False # Cờ nội bộ để theo dõi trạng thái tải

    def _construct_filepath(self, data_type: str) -> str:
        filename = f"{data_type}_{self.battery_id}.csv"
        return os.path.join(self.data_dir, filename)

    def load_data(self) -> bool:
        self._data_loaded = False # Reset trạng thái
        error_occurred = False
        data_types = ['capacity', 'charge', 'discharge']
        dfs = {}

        print(f"--- Loading data for battery: {self.battery_id} ---")
        for data_type in data_types:
            filepath = self._construct_filepath(data_type)
            try:
                if os.path.exists(filepath):
                    dfs[data_type] = pd.read_csv(filepath)
                    # Xóa cột 'Unnamed: 0' nếu có
                    if 'Unnamed: 0' in dfs[data_type].columns:
                        dfs[data_type] = dfs[data_type].drop(columns=['Unnamed: 0'])
                    print(f"  Successfully loaded: {os.path.basename(filepath)}")
                else:
                    print(f"  Warning: File not found - {filepath}. DataFrame set to None.")
                    dfs[data_type] = None
            except pd.errors.EmptyDataError:
                print(f"  Warning: File is empty - {filepath}. DataFrame set to None.")
                dfs[data_type] = None
            except Exception as e:
                print(f"  ERROR: Failed to read file {filepath} - {e}")
                dfs[data_type] = None
                error_occurred = True 

        self.capacity_df = dfs.get('capacity')
        self.charge_df = dfs.get('charge')
        self.discharge_df = dfs.get('discharge')

        # Đánh dấu là đã tải nếu không có lỗi đọc file nghiêm trọng
        if not error_occurred:
            self._data_loaded = True
            print(f"--- Data loading process completed for {self.battery_id} ---")
            return True
        else:
            print(f"--- Data loading process failed due to errors for {self.battery_id} ---")
            return False


    def get_raw_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        if not self._data_loaded:
            print("Data not previously loaded or load failed. Attempting load via get_raw_data()...")
            self.load_data()
        return self.capacity_df, self.charge_df, self.discharge_df

    def is_data_loaded(self) -> bool:
        return self._data_loaded