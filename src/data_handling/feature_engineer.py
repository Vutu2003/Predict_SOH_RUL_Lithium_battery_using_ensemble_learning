# Clas BatteryFeatureEngineer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm # Optional: for progress bar inside methods
from typing import Optional, Tuple, Dict, Any # For type hinting


# --- Lớp thứ 2: BatteryFeatureEngineer ---

class BatteryFeatureEngineer:
    DEFAULT_CONFIG = {
        'eol_threshold_percentage': 0.7, # Ngưỡng EoL (70% dung lượng ban đầu)
        'cc_current_threshold_low': 1.4, # Ngưỡng dòng xác định pha CC (dòng >= ngưỡng này)
        'cv_voltage_threshold_high': 4.15,# Ngưỡng áp xác định pha CV (áp >= ngưỡng này)
        'cv_current_threshold_high': 1.4, # Ngưỡng dòng xác định pha CV (dòng < ngưỡng này)
    }

    def __init__(self,
                 battery_id: str,
                 capacity_df: pd.DataFrame,
                 charge_df: pd.DataFrame,
                 discharge_df: pd.DataFrame,
                 config: Optional[Dict[str, Any]] = None):
        self.battery_id = battery_id
        self.raw_capacity_df = capacity_df.copy()
        self.raw_charge_df = charge_df.copy()
        self.raw_discharge_df = discharge_df.copy()

        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        # Thuộc tính để lưu kết quả
        self.processed_data: Optional[pd.DataFrame] = None
        self.initial_capacity: Optional[float] = None
        self.cycle_at_eol: Optional[int] = None

        # Đảm bảo kiểu dữ liệu số cho các cột cần thiết
        self._ensure_numeric_types()


    def _ensure_numeric_types(self):
        print(f"  Ensuring numeric types for {self.battery_id}...")
        for df, cols in [
            (self.raw_capacity_df, ['cycle', 'capacity']),
            (self.raw_charge_df, ['cycle', 'time', 'voltage_measured', 'current_measured', 'temperature_measured']),
            (self.raw_discharge_df, ['cycle', 'time', 'voltage_measured', 'current_measured', 'temperature_measured'])
        ]:
             if df is not None:
                for col in cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                df.dropna(subset=cols, inplace=True)


    def _calculate_soh_rul(self) -> Optional[pd.DataFrame]:
        if self.raw_capacity_df is None or self.raw_capacity_df.empty:
            print(f"  Warning: Capacity data is missing or empty for {self.battery_id}. Cannot calculate SOH/RUL.")
            return None

        print(f"  Calculating SOH/RUL for {self.battery_id}...")
        df = self.raw_capacity_df.sort_values(by='cycle').reset_index(drop=True)

        if 'capacity' not in df.columns or df['capacity'].isnull().all():
             print(f"  Error: 'capacity' column missing or all NaN in capacity data for {self.battery_id}.")
             return None

        self.initial_capacity = df['capacity'].iloc[0]
        if pd.isna(self.initial_capacity) or self.initial_capacity <= 0:
             print(f"  Error: Invalid initial capacity ({self.initial_capacity}) for {self.battery_id}.")
             return None

        df['SOH'] = (df['capacity'] / self.initial_capacity) * 100

        eol_threshold_capacity = self.initial_capacity * self.config['eol_threshold_percentage']
        cycle_at_eol_series = df[df['capacity'] < eol_threshold_capacity]['cycle']

        if not cycle_at_eol_series.empty:
            self.cycle_at_eol = int(cycle_at_eol_series.min())
        else:
            # Nếu không đạt EoL, dùng chu kỳ cuối + 1
            self.cycle_at_eol = int(df['cycle'].max()) + 1
            print(f"    Note: Battery {self.battery_id} did not reach EoL threshold ({eol_threshold_capacity:.2f} Ah). Using {self.cycle_at_eol} for RUL.")

        df['RUL'] = (self.cycle_at_eol - df['cycle']).clip(lower=0)

        # Giữ lại các cột cần thiết
        return df[['cycle', 'capacity', 'SOH', 'RUL']].copy()

    def _engineer_discharge_features(self) -> Optional[pd.DataFrame]:
        if self.raw_discharge_df is None or self.raw_discharge_df.empty:
            print(f"  Warning: Discharge data missing or empty for {self.battery_id}. Skipping discharge features.")
            return None

        print(f"  Engineering discharge features for {self.battery_id}...")
        features_list = []
        required_cols = ['voltage_measured', 'temperature_measured', 'time']
        if not all(col in self.raw_discharge_df.columns for col in required_cols):
             print(f"  Error: Missing required columns in discharge data for {self.battery_id}. Expected: {required_cols}")
             return None

        grouped = self.raw_discharge_df.groupby('cycle')
        for cycle, group in tqdm(grouped, desc=f"  Discharge {self.battery_id}", leave=False):
            group = group.sort_values(by='time')
            cycle_features = {'cycle': cycle}

            voltage = group['voltage_measured']
            temperature = group['temperature_measured']

            if not voltage.empty:
                cycle_features['Discharge_V_median'] = voltage.median()
                cycle_features['Discharge_V_skew'] = voltage.skew()
            else:
                 cycle_features['Discharge_V_median'] = np.nan
                 cycle_features['Discharge_V_skew'] = np.nan

            if not temperature.empty:
                cycle_features['Discharge_T_delta'] = temperature.max() - temperature.min() if len(temperature) > 1 else 0
                cycle_features['Discharge_T_std'] = temperature.std() if len(temperature) > 1 else 0
            else:
                 cycle_features['Discharge_T_delta'] = np.nan
                 cycle_features['Discharge_T_std'] = np.nan

            features_list.append(cycle_features)

        if not features_list:
            return None

        df_features = pd.DataFrame(features_list)
        return df_features.set_index('cycle')


    def _engineer_charge_features(self) -> Optional[pd.DataFrame]:
        if self.raw_charge_df is None or self.raw_charge_df.empty:
            print(f"  Warning: Charge data missing or empty for {self.battery_id}. Skipping charge features.")
            return None

        print(f"  Engineering charge features for {self.battery_id}...")
        features_list = []
        required_cols = ['voltage_measured', 'current_measured', 'temperature_measured', 'time']
        if not all(col in self.raw_charge_df.columns for col in required_cols):
             print(f"  Error: Missing required columns in charge data for {self.battery_id}. Expected: {required_cols}")
             return None

        # Lấy ngưỡng từ config
        cc_thr_low = self.config['cc_current_threshold_low']
        cv_v_thr_high = self.config['cv_voltage_threshold_high']
        cv_c_thr_high = self.config['cv_current_threshold_high']

        grouped = self.raw_charge_df.groupby('cycle')
        for cycle, group in tqdm(grouped, desc=f"  Charge {self.battery_id}", leave=False):
            group = group.sort_values(by='time').reset_index(drop=True)
            cycle_features = {'cycle': cycle}

            temperature = group['temperature_measured']
            if not temperature.empty and len(temperature) > 1:
                 cycle_features['Charge_T_std'] = temperature.std()
            else:
                 cycle_features['Charge_T_std'] = np.nan

            # Xác định pha CC và CV
            cc_phase = group[(group['current_measured'] >= cc_thr_low) & (group['voltage_measured'] < cv_v_thr_high)]
            cv_phase = group[(group['voltage_measured'] >= cv_v_thr_high) & (group['current_measured'] < cv_c_thr_high)]

            # Tính Time_CC_phase
            if not cc_phase.empty and len(cc_phase['time']) > 1:
                cycle_features['Time_CC_phase'] = cc_phase['time'].iloc[-1] - cc_phase['time'].iloc[0]
            else:
                cycle_features['Time_CC_phase'] = 0

            # Tính Time_CV_phase và CV_I_end
            if not cv_phase.empty:
                if len(cv_phase['time']) > 1:
                     cycle_features['Time_CV_phase'] = cv_phase['time'].iloc[-1] - cv_phase['time'].iloc[0]
                elif len(cv_phase['time']) == 1:
                      # Nếu chỉ có 1 điểm CV, ước tính thời gian từ đó đến hết
                      time_start_cv = cv_phase['time'].iloc[0]
                      time_end_charge = group['time'].max()
                      cycle_features['Time_CV_phase'] = time_end_charge - time_start_cv if time_end_charge > time_start_cv else 0
                else:
                     cycle_features['Time_CV_phase'] = 0
                cycle_features['CV_I_end'] = cv_phase['current_measured'].iloc[-1]
            else:
                cycle_features['Time_CV_phase'] = 0
                cycle_features['CV_I_end'] = np.nan # Không có pha CV thì không có dòng cuối CV

            features_list.append(cycle_features)

        if not features_list:
            return None

        df_features = pd.DataFrame(features_list)
        return df_features.set_index('cycle')

    def _merge_features(self,
                        capacity_soh_rul_df: Optional[pd.DataFrame],
                        discharge_features_df: Optional[pd.DataFrame],
                        charge_features_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if capacity_soh_rul_df is None:
            print(f"  Error: Cannot merge features for {self.battery_id} because SOH/RUL data is missing.")
            return None

        print(f"  Merging features for {self.battery_id}...")
        # Bắt đầu với df chứa SOH/RUL (có cột 'cycle')
        merged_df = capacity_soh_rul_df

        if discharge_features_df is not None:
            # Merge discharge features (index là 'cycle')
            merged_df = pd.merge(merged_df, discharge_features_df, on='cycle', how='left')
        else:
            print(f"Note: No discharge features to merge for {self.battery_id}.")

        if charge_features_df is not None:
            # Merge charge features (index là 'cycle')
            merged_df = pd.merge(merged_df, charge_features_df, on='cycle', how='left')
        else:
             print(f"Note: No charge features to merge for {self.battery_id}.")

        return merged_df

    def _handle_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """(Private) Xử lý NaN sau khi merge."""
        if df is None:
            return None
        print(f"  Handling NaNs for {self.battery_id}...")
        # Sử dụng bfill rồi ffill
        df_filled = df.fillna(method='bfill').fillna(method='ffill')
        # Kiểm tra lại nếu vẫn còn NaN (ví dụ: nếu toàn bộ cột là NaN)
        if df_filled.isnull().any().any():
             print(f"    Warning: NaNs still present after fillna for {self.battery_id}. Check data.")
             # Có thể thêm logic fill bằng giá trị cố định (0 hoặc mean/median) ở đây nếu cần
             # df_filled = df_filled.fillna(0) # Ví dụ
        return df_filled

    def process(self) -> 'BatteryFeatureEngineer':
        print(f"--- Starting Feature Engineering process for {self.battery_id} ---")
        # 1. Tính SOH/RUL
        capacity_soh_rul_df = self._calculate_soh_rul()

        # 2. Tính feature xả
        discharge_features_df = self._engineer_discharge_features()

        # 3. Tính feature sạc
        charge_features_df = self._engineer_charge_features()

        # 4. Merge features
        merged_df = self._merge_features(capacity_soh_rul_df, discharge_features_df, charge_features_df)

        # 5. Xử lý NaN cuối cùng
        processed_df_final = self._handle_nans(merged_df)

        # 6. Thêm cột battery_id
        if processed_df_final is not None:
            processed_df_final['battery_id'] = self.battery_id
            # Đảm bảo cột cycle tồn tại
            if 'cycle' not in processed_df_final.columns and processed_df_final.index.name == 'cycle':
                 processed_df_final = processed_df_final.reset_index()
            elif 'cycle' not in processed_df_final.columns:
                 print("  Error: 'cycle' column is missing after processing.")
                 self.processed_data = None
                 return self
        else:
             print(f"  Error: Feature processing failed for {self.battery_id}. Result is None.")


        self.processed_data = processed_df_final
        print(f"--- Feature Engineering process finished for {self.battery_id} ---")
        return self

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Trả về DataFrame chứa các feature đã được xử lý.

        Returns:
            Optional[pd.DataFrame]: DataFrame kết quả hoặc None nếu chưa xử lý hoặc lỗi.
        """
        if self.processed_data is None:
            print("Warning: Data has not been processed yet. Call process() first.")
        return self.processed_data
    
    # --- Các hàm vẽ đồ thị (Tùy chọn) ---
    def plot_feature_vs_cycle(self, feature_name: str, ax=None):
        """Vẽ đồ thị của một feature theo chu kỳ."""
        if self.processed_data is None or feature_name not in self.processed_data.columns:
            print(f"Error: Cannot plot. Data not processed or feature '{feature_name}' not found.")
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            show_plot = True
        else:
            show_plot = False # Không hiển thị nếu dùng subplot ngoài

        sns.lineplot(x='cycle', y=feature_name, data=self.processed_data, ax=ax, label=self.battery_id)
        ax.set_title(f'{feature_name} vs Cycle for {self.battery_id}')
        ax.set_xlabel('Cycle')
        ax.set_ylabel(feature_name)
        ax.grid(True)
        ax.legend()

        if show_plot:
            plt.show()

    def plot_soh_rul(self, ax_soh=None, ax_rul=None):
        """Vẽ đồ thị SOH và RUL theo chu kỳ."""
        if self.processed_data is None:
             print("Error: Cannot plot SOH/RUL. Data not processed.")
             return

        fig_created = False
        if ax_soh is None and ax_rul is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax_soh = ax1
            ax_rul = ax2
            fig.suptitle(f"SOH and RUL for {self.battery_id}")
            fig_created = True
        elif ax_soh is None or ax_rul is None:
             print("Warning: Provide both ax_soh and ax_rul or neither.")
             # Có thể tạo figure mới nếu chỉ cung cấp 1 ax
             return


        if 'SOH' in self.processed_data.columns:
             sns.lineplot(x='cycle', y='SOH', data=self.processed_data, ax=ax_soh, label=self.battery_id)
             ax_soh.set_title('SOH vs Cycle')
             ax_soh.set_ylabel('SOH (%)')
             ax_soh.grid(True)
             ax_soh.legend()
        else:
             ax_soh.set_title('SOH Data Missing')

        if 'RUL' in self.processed_data.columns:
             sns.lineplot(x='cycle', y='RUL', data=self.processed_data, ax=ax_rul, label=self.battery_id)
             ax_rul.set_title('RUL vs Cycle')
             ax_rul.set_ylabel('RUL (cycles)')
             ax_rul.grid(True)
             ax_rul.legend()
        else:
             ax_rul.set_title('RUL Data Missing')

        if fig_created:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
    def get_all_name_feature(self):
        return ['SOH', 'RUL', 'capacity',
                'Discharge_V_median', 'Discharge_V_skew', 'Discharge_T_delta', 'Discharge_T_std',
                'Charge_T_std', 'Time_CC_phase', 'Time_CV_phase', 'CV_I_end']