#!/usr/bin/env python3

from typing import List, Iterable
from pathlib import Path
from sklearn.cluster import KMeans
from argparse import ArgumentParser
from functools import reduce
from cross_dataset_common import find_files, get_tissue_type
import pandas as pd
import json
import numpy as np


def get_dataset(dataset_directory: Path) -> str:
    return dataset_directory.parent.stem


def get_tile_id(file: Path) -> str:
    return file.stem[0:15]


def get_attribute(file: Path) -> str:
    return file.stem.split("_")[0]


def coalesce_columns(df: pd.DataFrame, data_file: str, on_columns: List[str]) -> pd.DataFrame:

    print('coalesce_columns called')

    assert on_columns[0] in df.columns

    column_name = 'protein_' + data_file.split('_')[-1][:-4]
    dapi_columns = [column for column in df.columns if 'DAPI' in column]

    json_list = []

    for i, row in df.iterrows():
        protein_dict = {str(column): str(row[column]) for column in on_columns}
        json_item = str(json.dumps(protein_dict))
        json_list.append(json_item)

    df[column_name] = pd.Series(json_list, dtype=object)

    df.drop(dapi_columns, axis=1, inplace=True)

    df.drop(on_columns, axis=1, inplace=True)

    return df.copy()


def cluster_and_coalesce(df: pd.DataFrame, on_columns: List[str], data_file: str) -> pd.DataFrame:
    print('cluster and coalesce called')
    cluster_column_header = 'cluster_by_' + data_file.split('_')[-1][:-4]

    data = df[on_columns].to_numpy()
    cluster_assignments = KMeans(n_clusters=6, random_state=0).fit(data)
    df[cluster_column_header] = pd.Series(cluster_assignments.labels_, dtype=object)
    return coalesce_columns(df, data_file, on_columns)


def stitch_dfs(data_file: str, dataset_directory: Path, nexus_token: str) -> pd.DataFrame:
    print('stitch_df called')
    modality = 'codex'
    dataset = get_dataset(dataset_directory)
    tissue_type = get_tissue_type(dataset, nexus_token)

    csv_files = list(find_files(dataset_directory, data_file))

    tile_dfs = [pd.read_csv(csv_file, dtype=object) for csv_file in csv_files]
    tile_ids = [get_tile_id(csv_file) for csv_file in csv_files]

    for i, tile_df in enumerate(tile_dfs):
        tile_df['ID'] = tile_df['ID'].astype(str)
        tile_df['tile'] = tile_ids[i]
        tile_df['cell_id'] = modality + "-" + dataset + "-" + tile_df['tile'] + "-" + tile_df['ID']
        if data_file == '**cell_cluster.csv':
            for column in tile_df.columns:
                if column != 'ID' and column != 'tile' and column != 'cell_id':
                    tile_df[column] = modality + "-" + dataset + "-" + tile_df['tile'] + "-" + tile_df[column].astype(
                        str)
                    print(tile_df[column].unique())
        tile_df.drop(['ID', 'tile'], axis=1, inplace=True)

    stitched_df = reduce(outer_join, tile_dfs)

    if data_file != '**cell_shape.csv':
        protein_columns = [column for column in stitched_df.columns if column != 'cell_id' and 'DAPI' not in column]

        stitched_df = cluster_and_coalesce(stitched_df, protein_columns, data_file)

    stitched_df['dataset'] = dataset
    stitched_df['tissue_type'] = tissue_type
    stitched_df['modality'] = modality

    return stitched_df


def outer_join(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    return df_1.merge(df_2, how='outer')


def get_dataset_df(dataset_directory: Path, nexus_token: str) -> pd.DataFrame:

    per_cell_data_files = ['**cell_channel_covar.csv', '**cell_channel_mean.csv',
                           '**cell_channel_total.csv']

    stitched_dfs = [stitch_dfs(data_file, dataset_directory, nexus_token) for data_file in per_cell_data_files]

    dataset_df = reduce(outer_join, stitched_dfs)

    return dataset_df


def get_group_df(modality_df: pd.DataFrame) -> pd.DataFrame:
    group_columns = [column for column in modality_df.columns if 'cluster' in column or 'tissue' in column]
    group_dict_list = []

    for group_type in group_columns:
        for group_id in modality_df[group_type].unique():
            if type(group_id) == str or not np.isnan(group_id) :
                grouping_df = modality_df[modality_df[group_type] == group_id].copy()
                cell_ids = list(grouping_df['cell_id'].unique())
                group_dict_list.append({'group_type': group_type, 'group_id': str(group_id), 'cells': cell_ids})

    return pd.DataFrame(group_dict_list)


def main(nexus_token: str, output_directories: List[Path]):
    output_directories = [output_directory / Path('sprm_outputs') for output_directory in output_directories]
    dataset_dfs = [get_dataset_df(dataset_directory, nexus_token) for dataset_directory in output_directories]
    modality_df = pd.concat(dataset_dfs)
    group_df = get_group_df(modality_df)

    with pd.HDFStore('codex.hdf5') as store:
        store.put('cell', modality_df, format='t')
        store.put('group', group_df)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('nexus_token', type=str)
    p.add_argument('data_directories', type=Path, nargs='+')
    args = p.parse_args()

    main(args.nexus_token, args.data_directories)
