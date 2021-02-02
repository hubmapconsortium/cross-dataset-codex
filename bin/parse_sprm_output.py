#!/usr/bin/env python3

from argparse import ArgumentParser
from functools import reduce
from pathlib import Path
from typing import List

import pandas as pd
from cross_dataset_common import find_files, get_tissue_type, create_minimal_dataset
from hubmap_cell_id_gen_py import get_spatial_cell_id


def get_dataset(dataset_directory: Path) -> str:
    return dataset_directory.stem


def get_tile_id(file: Path) -> str:
    return file.stem[0:15]


def get_attribute(file: Path) -> str:
    return file.stem.split("_")[0]


def flatten_df(df: pd.DataFrame, statistic:str) -> pd.DataFrame:
    on_columns = [column for column in df.columns if column != 'cell_id' and 'DAPI' not in column]
    dict_list = [{'q_var_id': column, 'q_cell_id': i, 'statistic':statistic, 'value': df.at[i, column]} for column in on_columns for i in
                 df.index]
    return pd.DataFrame(dict_list)


def stitch_dfs(data_file: str, dataset_directory: Path, nexus_token: str) -> pd.DataFrame:
    modality = 'codex'
    dataset = get_dataset(dataset_directory)
    tissue_type = get_tissue_type(dataset, nexus_token)

    csv_files = list(find_files(dataset_directory, data_file))

    tile_ids_and_dfs = [(get_tile_id(csv_file), pd.read_csv(csv_file)) for csv_file in csv_files]

    for tile_id, tile_df in tile_ids_and_dfs:
        tile_df['ID'] = tile_df['ID'].astype(str)
        tile_df['tile'] = tile_id
        tile_df['cell_id'] = pd.Series(
            [get_spatial_cell_id(dataset, tile_id, mask_index) for mask_index in tile_df['ID']])
        tile_df.drop(['ID'], axis=1, inplace=True)

    stitched_df = reduce(outer_join, [id_and_df[1] for id_and_df in tile_ids_and_dfs])

    stitched_df['dataset'] = dataset
    stitched_df['organ_name'] = tissue_type
    stitched_df['modality'] = modality

    return stitched_df


def outer_join(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    return df_1.merge(df_2, how='outer')


def get_dataset_dfs(dataset_directory: Path, nexus_token: str) -> (pd.DataFrame, pd.DataFrame):
    per_cell_data_files = ['**cell_channel_covar.csv', '**cell_channel_mean.csv']

    stitched_dfs = [stitch_dfs(data_file, dataset_directory, nexus_token) for data_file in per_cell_data_files]
    statistics = [file.split('_')[0] for file in per_cell_data_files]
    stitched_dfs_and_stats = zip(stitched_dfs, statistics)

    quant_dfs = [flatten_df(df, stat) for df, stat in stitched_dfs_and_stats]

    quant_df = pd.concat(quant_dfs)

    dataset_df = reduce(outer_join, stitched_dfs)

    return dataset_df, quant_df


def main(nexus_token: str, dataset_directories: List[Path]):
    dataset_dfs = []
    quant_dfs = []
    for dataset_directory in dataset_directories:
        dataset_df, quant_df = get_dataset_dfs(dataset_directory, nexus_token)
        dataset_dfs.append(dataset_df)
        quant_dfs.append(dataset_df)

    cell_df = pd.concat(dataset_dfs)
    quant_df = pd.concat(quant_dfs)

    with pd.HDFStore('codex.hdf5') as store:
        store.put('cell', cell_df)

    create_minimal_dataset(cell_df, quant_df, modality='codex')

    quant_df.to_csv('codex.csv')

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('nexus_token', type=str)
    p.add_argument('data_directories', type=Path, nargs='+')
    args = p.parse_args()

    main(args.nexus_token, args.data_directories)
