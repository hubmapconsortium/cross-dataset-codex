#!/usr/bin/env python3

from argparse import ArgumentParser
from functools import reduce
from pathlib import Path
from typing import List, Dict, Iterable

import pandas as pd
from hubmap_cell_id_gen_py import get_spatial_cell_id
from cross_dataset_common import find_files, get_tissue_type, create_minimal_dataset

from concurrent.futures import ThreadPoolExecutor

def get_dataset(dataset_directory: Path) -> str:
    return dataset_directory.stem


def get_tile_id(file: Path) -> str:
    return str(file.stem).split('.')[0]


def get_cluster_assignments(dataset_directory, tile):
    cluster_file = dataset_directory / Path(f"sprm_outputs/{tile}.ome.tiff-cell_cluster.csv")
    cluster_df = pd.read_csv(cluster_file)

    cluster_assignments_list = [",".join([f"KMeans-{column.split('[')[1].split(']')[0]}-{dataset_directory.stem}-{tile}-" \
                                          f"{cluster_df.at[i, column]}" for column in cluster_df.columns[1:]]) for i in
                                cluster_df.index]

    return cluster_assignments_list


def flatten_df(df: pd.DataFrame, statistic: str) -> pd.DataFrame:
    on_columns = [column for column in df.columns if column not in ['cell_id', 'clusters', 'dataset', 'organ', 'tile',
                                                                    'modality'] and 'DAPI' not in column]
    dict_list = [
        {'q_var_id': column, 'q_cell_id': df.at[i, 'cell_id'], 'statistic': statistic, 'value': df.at[i, column]}
        for i in df.index for column in on_columns]

    return dict_list


def stitch_dfs(data_file: str, dataset_directory: Path, nexus_token: str) -> pd.DataFrame:
    modality = 'codex'
    dataset = get_dataset(dataset_directory)
    tissue_type = get_tissue_type(dataset, nexus_token)

    csv_files = [file for file in find_files(dataset_directory, data_file) if 'R' in file.stem]

    tile_ids_and_dfs = [(get_tile_id(csv_file), pd.read_csv(csv_file)) for csv_file in csv_files]

    for id_and_df in tile_ids_and_dfs:
        tile_id = id_and_df[0]
        tile_df = id_and_df[1]
        tile_df['ID'] = tile_df['ID'].astype(str)
        tile_df['tile'] = tile_id
        tile_df['cell_id'] = pd.Series(
            [get_spatial_cell_id(dataset, tile_id, mask_index) for mask_index in tile_df['ID']], index=tile_df.index)
        tile_df.drop(['ID'], axis=1, inplace=True)
        if data_file == '**cell_channel_total.csv':  # If this is the third set of CSVs for this dataset
            tile_df['clusters'] = pd.Series(get_cluster_assignments(dataset_directory, tile_id), index=tile_df.index)

    stitched_df = concat_dfs([id_and_df[1] for id_and_df in tile_ids_and_dfs])

    stitched_df['dataset'] = pd.Series([dataset for i in stitched_df.index], index=stitched_df.index)
    stitched_df['organ'] = tissue_type
    stitched_df['modality'] = modality

    return stitched_df


def concat_dfs(dfs: List[pd.DataFrame]):
    new_df_list = []
    for df in dfs:
        new_df_list.extend(df.to_dict(orient='records'))
    return pd.DataFrame(new_df_list)


def outer_join(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    return df_1.merge(df_2, how='outer')


def get_dataset_dfs(dataset_directory: Path, nexus_token: str=None) -> (pd.DataFrame, pd.DataFrame):
    per_cell_data_files = ['**cell_channel_covar.csv', '**cell_channel_mean.csv', '**cell_channel_total.csv']

    stitched_dfs = [stitch_dfs(data_file, dataset_directory, nexus_token) for data_file in per_cell_data_files]
    statistics = [file.split('_')[2][:-4] for file in per_cell_data_files]
    df_stats = {statistics[i]:stitched_dfs[i] for i in range(len(stitched_dfs))}

    quant_df_lists = []

    for stat in df_stats:
        dict_list = flatten_df(df_stats[stat], stat)
        quant_df_lists.extend(dict_list)

    quant_df = pd.DataFrame(quant_df_lists)

    return dataset_df, quant_df


def main(nexus_token: str, known_hosts_file: Path, dataset_directories: List[Path]):
    nexus_token = None if nexus_token == "None" else nexus_token

    with ThreadPoolExecutor(max_workers=len(dataset_directories)) as e:
        df_pairs = list(e.map(get_dataset_dfs, dataset_directories))

    dataset_dfs = [df_pair[0] for df_pair in df_pairs]
    quant_dfs = [df_pair[1] for df_pair in df_pairs]

    quant_df = concat_dfs(quant_dfs)
    cell_df = concat_dfs(dataset_dfs)

    cell_df = cell_df[['cell_id', 'dataset', 'modality', 'organ', 'tile', 'clusters']]

    with pd.HDFStore('codex.hdf5') as store:
        store.put('cell', cell_df)

    create_minimal_dataset(cell_df, quant_df, modality='codex')

    quant_df.to_csv('codex.csv')

    tar_zip_scp("codex", known_hosts_file)

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('nexus_token', type=str)
    p.add_argument('known_hosts_file', type=Path)
    p.add_argument('data_directories', type=Path, nargs='+')
    args = p.parse_args()

    main(args.nexus_token, args.known_hosts_file, args.data_directories)
