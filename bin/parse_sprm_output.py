#!/usr/bin/env python3

from argparse import ArgumentParser
from functools import reduce
from pathlib import Path
from typing import List, Dict, Iterable

import pandas as pd
from hubmap_cell_id_gen_py import get_spatial_cell_id
from cross_dataset_common import find_files, get_tissue_type, create_minimal_dataset

from concurrent.futures import ThreadPoolExecutor

def outer_join(df1:pd.DataFrame, df2:pd.DataFrame):
    return df1.merge(df2, how='outer')

def get_image_regions(keys):
    return set([key.split('/')[-1] for key in keys])

def get_quantification_keys(statistic, cell_region, image_region):
    return f"/{statistic}/channel/{cell_region}/expressions.ome.tiff/stitched/{image_region}"

def get_cluster_keys(image_region):
    return f"/cluster/cells/expressions.ome.tiff/stitched/{image_region}"

def get_dataset(dataset_directory: Path) -> str:
    return dataset_directory.stem

def get_cluster_assignments(cluster_df, region):

    cluster_assignments_list = [",".join([f"KMeans-{column.split('[')[1].split(']')[0]}-{dataset_directory.stem}-{region}-" \
                                          f"{cluster_df.at[i, column]}" for column in cluster_df.columns[1:]]) for i in
                                cluster_df.index]

    return cluster_assignments_list


def flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    on_columns = [column for column in df.columns if column not in ['cell_id', 'clusters', 'dataset', 'organ', 'tile',
                                                                    'modality'] and 'DAPI' not in column]
    dict_list = [
        {'q_var_id': column.split('_')[2], 'q_cell_id': df.at[i, 'cell_id'], 'statistic': column.split('_')[1], 'region':
            column.split('_')[0], 'value': df.at[i, column]}
        for i in df.index for column in on_columns]

    return pd.DataFrame(dict_list)

def get_quantification_df(hdf_file, statistic, cell_region, image_region):
    quantification_key = get_quantification_keys(statistic, cell_region, image_region)
    quantification_df = pd.read_hdf(hdf_file, quantification_key)
    new_columns = [f"{cell_region}_{statistic}_{column}" if column != "ID" else column for column in
                   quantification_df.columns]
    quantification_df.columns = new_columns
    return quantification_df

def get_region_df(hdf_file, image_region):
    statistics = ["mean", "total", "covar"]
    cell_regions = ["cell", "nucleus"]
    cluster_key = get_cluster_keys(image_region)
    cluster_df = pd.read_hdf(hdf_file, cluster_key)
    quantification_dfs = [get_quantification_df(hdf_file, statistic, cell_region, image_region) for statistic in
                          statistics for cell_region in cell_regions]
    region_df = reduce(outer_join, quantification_dfs)
    region_df["clusters"] = pd.Series(get_cluster_assignments(cluster_df, image_region))
    region_df['cell_id'] = pd.Series(
        [get_spatial_cell_id(dataset, image_region, mask_index) for mask_index in region_df['ID']], index=region_df.index)

    return region_df


def annotate_df(dataset_directory: Path, nexus_token: str) -> pd.DataFrame:
    modality = 'codex'
    dataset = get_dataset(dataset_directory)
    tissue_type = get_tissue_type(dataset, nexus_token)

    hdf_file = dataset_directory / Path('sprm-outputs/out.hdf5')
    hdf_keys = hdf_file.keys()
    image_regions = get_image_regions(hdf_keys)

    region_dfs = [get_region_df(hdf_file, region) for region in image_regions]

    dataset_df = concat_dfs(region_dfs) if len(region_dfs) > 1 else region_dfs[0]

    dataset_df['dataset'] = dataset
    dataset_df['organ'] = tissue_type
    dataset_df['modality'] = modality

    return stitched_df


def concat_dfs(dfs: List[pd.DataFrame]):
    new_df_list = []
    for df in dfs:
        new_df_list.extend(df.to_dict(orient='records'))
    return pd.DataFrame(new_df_list)


def get_dataset_dfs(dataset_directory: Path, nexus_token: str=None) -> (pd.DataFrame, pd.DataFrame):

    dataset_df = annotate_df(dataset_directory, nexus_token)
    quant_df = flatten_df(dataset_df)

    return dataset_df, quant_df


def main(nexus_token: str, dataset_directories: List[Path]):
    print("Main called")
    nexus_token = None if nexus_token == "None" else nexus_token

    with ThreadPoolExecutor(max_workers=len(dataset_directories)) as e:
        df_pairs = list(e.map(get_dataset_dfs, dataset_directories))

    dataset_dfs = [df_pair[0] for df_pair in df_pairs]
    quant_dfs = [df_pair[1] for df_pair in df_pairs]

    quant_df = concat_dfs(quant_dfs)
    cell_df = concat_dfs(dataset_dfs)

    cell_df = cell_df[['cell_id', 'dataset', 'modality', 'organ', 'clusters']]

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
