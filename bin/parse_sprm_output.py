#!/usr/bin/env python3

from argparse import ArgumentParser
from functools import reduce
from pathlib import Path
from typing import List, Dict, Iterable
from scipy.io import mmread
from scipy.sparse import coo_matrix

import anndata
import pandas as pd
from hubmap_cell_id_gen_py import get_spatial_cell_id
from cross_dataset_common import find_files, get_tissue_type, create_minimal_dataset, precompute_dataset_percentages, precompute_values_series

from concurrent.futures import ThreadPoolExecutor

ADJACENCY_MATRIX_PATH_PATTERN = 'reg1_stitched_expressions.ome.tiff_AdjacencyMatrix.mtx'

def get_adjacency_adata(adjacency_file):
    region = get_reg_id(file)
    dataset_uuid = adjacency_file.parent.parent.stem
    matrix = mmread(adjacency_file)
    cells = [f"{dataset_uuid}-{region}-{rows[i]}"for i in range(matrix.shape[0])]
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame(index=cells)
    cell_by_cell_adata = anndata.AnnData(X=matrix, obs=obs, var=var)
    return cell_by_cell_adata

def get_dataset_adjacency_adata(dataset_directory):
    adjacency_files = find_files(dataset_directory, ADJACENCY_MATRIX_PATH_PATTERN)
    if len(adjacency_files) == 0:
        return None
    adjacency_adatas = [get_adjacency_adata(file) for file in adjacency_files]
    if len(adjacency_adatas) == 1:
        return adjacency_adatas[0]
    dataset_adjacency_adata = adjacency_adatas[0].concatenate(adjacency_adatas[1:])
    return dataset_adjacency_adata

def make_anndata(quant_df):

    on_columns = [column for column in df.columns if column not in ['cell_id', 'clusters', 'dataset', 'organ', 'tile',
                                                                    'modality'] and 'DAPI' not in column]
    var = pd.DataFrame(index=on_columns)
    obs = pd.DataFrame(index=quant_dfs_and_stats[0][0].index)
    adata = anndata.AnnData(var=var, obs=obs)

    for i in df.index:
        for column in on_columns:
            adata.X[i][column] = quant_df.at[i, column]

    return adata


def get_dataset(dataset_directory: Path) -> str:
    return dataset_directory.stem


def get_tile_id(file: Path) -> str:
    return str(file.stem).split('.')[0]

def get_reg_id(file: str) -> str:
    return str(file.stem).split('_')[0]

def get_cluster_assignments_tile(dataset_directory, tile):
    cluster_file = dataset_directory / Path(f"sprm_outputs/{tile}.ome.tiff-cell_cluster.csv")
    cluster_df = pd.read_csv(cluster_file)

    cluster_assignments_list = [",".join([f"KMeans-{column.split('[')[1].split(']')[0]}-{dataset_directory.stem}-{tile}-" \
                                          f"{cluster_df.at[i, column]}" for column in cluster_df.columns[1:]]) for i in
                                cluster_df.index]

    return cluster_assignments_list

def get_cluster_assignments_stitched(dataset_directory, region):
    cluster_file = dataset_directory / Path(f"sprm_outputs/{region}_stitched_expressions.ome.tiff-cell_cluster.csv")
    cluster_df = pd.read_csv(cluster_file)

    cluster_assignments_list = [",".join([f"KMeans-{column.split('[')[1].split(']')[0]}-{dataset_directory.stem}-{tile}-" \
                                          f"{cluster_df.at[i, column]}" for column in cluster_df.columns[1:]]) for i in
                                cluster_df.index]

    return cluster_assignments_list


def stitch_dfs_tile(data_file: str, dataset_directory: Path, nexus_token: str) -> pd.DataFrame:
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


def stitch_dfs_stitched(data_file: str, dataset_directory: Path, nexus_token: str) -> pd.DataFrame:
    modality = 'codex'
    dataset = get_dataset(dataset_directory)
    tissue_type = get_tissue_type(dataset, nexus_token)

    csv_files = [file for file in find_files(dataset_directory, data_file) if 'reg' in file.stem]

    reg_ids_and_dfs = [(get_reg_id(file), pd.read_csv(file)) for file in csv_files]

    for id_and_df in reg_ids_and_dfs:
        tile_id = id_and_df[0]
        tile_df = id_and_df[1]
        tile_df['tile'] = tile_id
        tile_df['cell_id'] = pd.Series(
            [get_spatial_cell_id(dataset, tile_id, mask_index) for mask_index in tile_df['ID']], index=tile_df.index)
        tile_df.drop(['ID'], axis=1, inplace=True)
        if data_file == '**cell_channel_total.csv':  # If this is the third set of CSVs for this dataset
            tile_df['clusters'] = pd.Series(get_cluster_assignments(dataset_directory, tile_id), index=tile_df.index)

    stitched_df = concat_dfs([id_and_df[1] for id_and_df in reg_ids_and_dfs])

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


def get_dataset_dfs_tile(dataset_directory: Path, nexus_token: str=None) -> (pd.DataFrame, anndata.AnnData):
    per_cell_data_files = ['**cell_channel_covar.csv', '**cell_channel_mean.csv', '**cell_channel_total.csv']

    stitched_dfs = [stitch_dfs_tile(data_file, dataset_directory, nexus_token) for data_file in per_cell_data_files]

    dataset_df = reduce(outer_join, stitched_dfs)
    adata = make_anndata(stitched_dfs[1])

    dataset_df = dataset_df[['cell_id', 'dataset', 'modality', 'organ', 'tile', 'clusters']]

    return dataset_df, adata


def get_dataset_dfs_stitched(dataset_directory: Path, nexus_token: str=None) -> (pd.DataFrame, anndata.AnnData):
    per_cell_data_files = ['**cell_channel_covar.csv', '**cell_channel_mean.csv', '**cell_channel_total.csv']

    stitched_dfs = [stitch_dfs_stitched(data_file, dataset_directory, nexus_token) for data_file in per_cell_data_files]

    adata = make_anndata(stitched_dfs[1])

    quant_df_lists = []

    for stat in df_stats:
        dict_list = flatten_df(df_stats[stat], stat)
        quant_df_lists.extend(dict_list)

    dataset_df = reduce(outer_join, stitched_dfs)
    dataset_df = dataset_df[['cell_id', 'dataset', 'modality', 'organ', 'tile', 'clusters']]

    adata = make_anndata(stitched_dfs[1])

    return dataset_df, adata

def get_dataset_dfs(dataset_directory, nexus_token=None):
    try:
        return get_dataset_dfs_stitched(dataset_directory, nexus_token)
    except FileNotFoundError:
        return get_dataset_dfs_tile(dataset_directory, nexus_token)

def main(nexus_token: str, dataset_directories: List[Path]):
    nexus_token = None if nexus_token == "None" else nexus_token

    with ThreadPoolExecutor(max_workers=len(dataset_directories)) as e:
        df_pairs = list(e.map(get_dataset_dfs, dataset_directories))

    dataset_dfs = [df_pair[0] for df_pair in df_pairs]
    adatas = [df_pair[1] for df_pair in df_pairs]
    print("Got dataset dfs and adatas")

    with ThreadPoolExecutor(max_workers=len(directories)) as e:
        percentage_dfs = e.map(precompute_dataset_percentages, adatas)

    print("Got percentage dfs")

    percentage_df = pd.concat(percentage_dfs)

    adata = adatas[0].concatenate(adatas[1:])
    cell_df = concat_dfs(dataset_dfs)

    print("Got cell df")

    values_series_dict = precompute_values_series(cell_df, adata)

    print("Got precomptued values")

    cell_df = cell_df[['cell_id', 'dataset', 'modality', 'organ', 'tile', 'clusters']]

    adjacency_adatas = [get_dataset_adjacency_adata(dataset_directory) for dataset_directory in dataset_directories if get_adjacency_adata(dataset_directory)]
    adjacency_adata = adjacency_adatas[0].concatenate(adjacency_adatas[1:])
    adjacency_adata.write('codex_adjacency.h5ad')

    print("Got adjacency data")

    with pd.HDFStore('codex.hdf5') as store:
        store.put('cell', cell_df)

    with pd.HDFStore('codex_precompute.hdf5') as store:
        for key in values_series_dict:
            store.put(key, values_series_dict[key])
        store.put('percentages', percentage_df)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('nexus_token', type=str)
    p.add_argument('data_directories', type=Path, nargs='+')
    p.add_argument("--enable-manhole", action="store_true")
    args = p.parse_args()

    if args.enable_manhole:
        import manhole

        manhole.install(activate_on="USR1")

    main(args.nexus_token, args.data_directories)
