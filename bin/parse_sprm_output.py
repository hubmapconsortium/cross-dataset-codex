import pandas as pd
import numpy as np
from typing import Iterable
from pathlib import Path
from os import walk
import sqlite3

def get_datasets():
    return

def get_metadata():
    return

def get_csv_files(partial_file_name: str, directory: Path)-> Iterable[Path]:
    for dirpath_str, dirnames, filenames in walk(directory):
        dirpath = Path(dirpath_str)
        for filename in filenames:
            filepath = dirpath / filename
            if filepath.match(partial_file_name):
                yield filepath

def get_tile_id(file: Path)-> str:
    return file.stem[0:15]

def get_attribute(file: Path)->str:
    return file.stem.split("_")[0]

def main(output_directory: Path):

    database_file = Path('codex.db')
    per_cell_data_files = ['**cell_shape.csv', '**cell_channel_covar.csv', '**cell_channel_mean.csv', '**cell_channel_total.csv']

    conn = sqlite3.connect(database_file)
    cur = conn.cursor()

    stitched_dfs = {}

for data_file in per_cell_data_files:

    csv_files = list(get_csv_files(data_file, directory))

    tile_dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    tile_ids = [get_tile_id(csv_file) for csv_file in csv_files]

    for i, tile_df in enumerate(tile_dfs):
        tile_df['ID'] = tile_df['ID'].astype(str)
        tile_df['tile'] = tile_ids[i] + "_"
        tile_df['cell_id'] = tile_df['tile'] + tile_df['ID']
        tile_df.drop('ID', axis=1)
        tile_df.drop('tile', axis=1)

    for tile_df in tile_dfs[1:]:
        tile_dfs[0] = tile_dfs[0].merge(tile_df, how='outer')
        old_columns = tile_df.columns
        new_columns = [column + get_attribute(csv_files[0]) for column in old_columns]
        new_columns[0] = 'cell_id'

        tile_dfs[0].columns = new_columns

        stitched_dfs[data_file] = tile_dfs[0]

    stitched_dfs_list = list(stitched_dfs.values())

    dataset_df = stitched_dfs_list[0]

    for stitched_df in stitched_dfs_list[1:]:
        dataset_df = dataset_df.merge(stitched_df, how='outer')

    dataset_df = dataset_df.set_index('cell_id')

    dataset_df.to_sql('codex', conn, if_exists='replace', index=True)

    conn.close()


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('output_directory', type=Path)
    args = p.parse_args()

    main(args.output_directory)
