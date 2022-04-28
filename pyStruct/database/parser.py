""" 
Parse the cfd snapshots into time-series
"""

from pathlib import Path
from tqdm import tqdm
import os 
import pandas as pd
import numpy as np


def extract_col_from_file(path: str, columns: list[str]):
    # use pandas to do file processing
    df = pd.read_csv(path)
    df = df.sort_values(by=['X (m)', 'Y (m)', 'Z (m)'])
    return {column: tuple(df[column]) for column in columns}

class TimeSeriesParser:
    def __init__(self, csv_paths: list[Path], dst_folder: Path):
        self.csv_paths = csv_paths
        self.dst_folder = dst_folder

        # Initialize the folder if it doesn't exist
        self.dst_folder.mkdir(parents=True, exist_ok=True)
    
    def _get_coordinates(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_paths[0])
        df = df.sort_values(by=['X (m)', 'Y (m)', 'Z (m)'])
        coord = df[['X (m)', 'Y (m)', 'Z (m)']]
        return coord
    
    def _get_columns(self) -> list[str]:
        df = pd.read_csv(self.csv_paths[0])
        return list(df.keys())
        
    def _update_coordinates(self, coord):
        save_to = self.dst_folder / "coord.csv"
        coord.to_csv(save_to, index=False) 
    
    def parse(self):
        # Get the columns from the csv file
        columns = self._get_columns() 
        place_holders_for_columns = {key:[] for key in columns}
        for csv_path in tqdm(self.csv_paths):
            columns_data = extract_col_from_file(csv_path, columns=columns)
            for column in columns:
                place_holders_for_columns[column].append(columns_data[column])
        for column in columns:
            place_holders_for_columns[column] = np.array(place_holders_for_columns[column])
        
    
        N_t, N_grid = place_holders_for_columns[column].shape
        for loc in range(N_grid):
            data = {column:place_holders_for_columns[column][:, loc] for column in columns}
            df = pd.DataFrame(data, columns=columns)
            save_to = self.dst_folder / f"loc_{loc}.csv"
            df.to_csv(save_to, index=False)

        coord = self._get_coordinates()
        self._update_coordinates(coord)
        



