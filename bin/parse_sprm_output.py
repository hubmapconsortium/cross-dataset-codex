#!/usr/bin/env python3

from argparse import ArgumentParser
from functools import reduce
from pathlib import Path
from typing import List, Dict

import pandas as pd
from cross_dataset_common import find_files, get_tissue_type, create_minimal_dataset
from hubmap_cell_id_gen_py import get_spatial_cell_id

def get_cluster_assignments(dataset_dfs, tile_ids)->dict:
    clusters_dict = {}
    for dataset_directory in dataset_dfs.keys():
        dataset_dict = {}
        dataset_df = dataset_dfs[dataset_directory]
        for tile_id in tile_ids[dataset_directory.stem]:
            tile_dict = {}
            cluster_file = dataset_directory / Path(f"sprm_outputs/{tile_id}.ome.tiff-cell_cluster.csv")
            cluster_df = pd.read_csv(cluster_file)
            datasets_list = dataset_df["dataset"].to_list()
            tiles_list = dataset_df["tile"].to_list()
            columns_lists = {column:cluster_df[column].to_list() for column in cluster_df.columns[:1]}
            for i in cluster_df.index:
                tile_dict[i + 1] = [f"KMeans-{column.split('[')[1].split(']')[0]}-{datasets_list[i]}-{tiles_list[i]}-" \
                         f"{columns_lists[column][i]}" for column in columns_lists]
            dataset_dict[tile_id] = tile_dict
        clusters_dict[dataset_directory.stem] = dataset_dict
    return clusters_dict

def get_cluster(cell_id, cluster_assignments):
    fields = cell_id.split('-')
    dataset = fields[0]
    tile_id = fields[1]
    original_index = int(fields[2])
    try:
        return cluster_assignments[dataset][tile_id][original_index]
    except:
        print(cell_id)
        if dataset not in cluster_assignments.keys():
            print(dataset)
            print(cluster_assignments.keys())
        elif tile_id not in cluster_assignments[dataset].keys():
            print(tile_id)
            print(cluster_assignments[dataset].keys())
        elif original_index not in cluster_assignments[dataset][tile_id].keys():
            print(original_index)
            print(cluster_assignments[dataset][tile_id].keys())
        return []

def assign_clusters(cell_df, cluster_assignments):
    all_cluster_assignments = [get_cluster(i, cluster_assignments) for i in cell_df.index]
    cell_df['clusters'] = pd.Series(all_cluster_assignments, index=cell_df.index)
    return cell_df

def get_dataset(dataset_directory: Path) -> str:
    return dataset_directory.stem

def get_all_tile_ids(dataset_directories: List[Path]) ->Dict:
    all_tile_ids = {}
    for dataset_directory in dataset_directories:
        csv_files = [file for file in find_files(dataset_directory, '**cell_cluster.csv') if 'R' in file.stem]
        tile_ids_set = set({})
        for file in csv_files:
            tile_ids_set.add(get_tile_id(file))
        all_tile_ids[dataset_directory.stem] = tile_ids_set

    return all_tile_ids

def get_tile_id(file: Path) -> str:
    return str(file.stem).split('.')[0]

def get_attribute(file: Path) -> str:
    return file.stem.split("_")[0]

def flatten_df(df: pd.DataFrame, statistic:str) -> pd.DataFrame:
    on_columns = [column for column in df.columns if column not in ['cell_id', 'clusters', 'dataset', 'organ', 'tile',
                                                                    'modality'] and 'DAPI' not in column]
    dict_list = []

    for i in df.index:
        for column in on_columns:
            temp_dict = {}
            temp_dict['q_var_id'] = column
            temp_dict['q_cell_id'] = df['cell_id'][i]
            temp_dict['statistic'] = statistic
            temp_dict['value'] = df['cell_id'][i]
            dict_list.append(temp_dict)

    dict_list = [{'q_var_id': column, 'q_cell_id': df['cell_id'][i], 'statistic':statistic, 'value': df[column][i]}
                 for i in df.index for column in on_columns]
    return pd.DataFrame(dict_list)


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
            [get_spatial_cell_id(dataset, tile_id, mask_index) for mask_index in tile_df['ID']])
        tile_df.drop(['ID'], axis=1, inplace=True)

    stitched_df = reduce(outer_join, [id_and_df[1] for id_and_df in tile_ids_and_dfs])

    stitched_df['dataset'] = pd.Series([dataset for i in stitched_df.index], index=stitched_df.index)
    stitched_df['organ'] = tissue_type
    stitched_df['modality'] = modality

    return stitched_df


def outer_join(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    return df_1.merge(df_2, how='outer')

def outer_join_on_index(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    return df_1.merge(df_2, how='outer', on='cell_id')


def get_dataset_dfs(dataset_directory: Path, nexus_token: str) -> (pd.DataFrame, pd.DataFrame):
    per_cell_data_files = ['**cell_channel_covar.csv', '**cell_channel_mean.csv', '**cell_channel_total.csv']

    stitched_dfs = [stitch_dfs(data_file, dataset_directory, nexus_token) for data_file in per_cell_data_files]
    statistics = [file.split('_')[2][:-4] for file in per_cell_data_files]
    stitched_dfs_and_stats = zip(stitched_dfs, statistics)

    quant_dfs = [flatten_df(df, stat) for df, stat in stitched_dfs_and_stats]

    quant_df = pd.concat(quant_dfs)

    stitched_dfs = [df.set_index('cell_id', drop=True, inplace=False) for df in stitched_dfs]
    dataset_df = reduce(outer_join_on_index, stitched_dfs)
#    dataset_df = pd.concat(stitched_dfs)

    return dataset_df, quant_df


def main(nexus_token: str, dataset_directories: List[Path]):
    dataset_dfs = {}
    quant_dfs = []
    for dataset_directory in dataset_directories:
        dataset_df, quant_df = get_dataset_dfs(dataset_directory, nexus_token)
        dataset_dfs[dataset_directory] = dataset_df
        quant_dfs.append(quant_df)

    tile_ids = get_all_tile_ids(dataset_directories)
    cluster_assignments = get_cluster_assignments(dataset_dfs, tile_ids)

    quant_df = pd.concat(quant_dfs)

    cell_df = pd.concat([dataset_df for dataset_df in dataset_dfs.values()])
    print(cell_df.index)

    cell_df = cell_df[['dataset', 'modality', 'organ_name', 'tile']]
    print(cell_df.index)
    cell_df = assign_clusters(cell_df, cluster_assignments)


    cell_df['cell_id'] = cell_df.index

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
