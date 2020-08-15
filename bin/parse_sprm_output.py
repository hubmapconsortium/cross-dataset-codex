import pandas as pd
import numpy as np
from typing import Iterable, List
from pathlib import Path
from os import walk, fspath
import sys
import json
from sklearn.cluster import KMeans
import yaml

def get_dataset(dataset_directory: Path)->str:
    return dataset_directory.stem

def get_tissue_type(dataset:str, token:str)->str:

    organ_dict = yaml.load(open('organ_types.yaml'), Loader=yaml.BaseLoader)

    dataset_query_dict = {
       "query": {
         "bool": {
           "must": [],
           "filter": [
             {
               "match_all": {}
             },
             {
               "exists": {
                 "field": "files.rel_path"
               }
             },
             {
               "match_phrase": {
                 "uuid": {
                   "query": dataset
                 },
               }

             }
           ],
           "should": [],
           "must_not": [
             {
               "match_phrase": {
                 "status": {
                   "query": "Error"
                 }
               }
             }
           ]
         }
       }
     }

    dataset_response = requests.post(
    'https://search-api.dev.hubmapconsortium.org/search',
    json = dataset_query_dict,
    headers = {'Authorization': 'Bearer ' + token})
    hits = dataset_response.json()['hits']['hits']

    for hit in hits:
        for ancestor in hit['_source']['ancestors']:
            if 'organ' in ancestor.keys():
                return organ_dict[ancestor['organ']]['description']

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

def coalesce_columns(df: pd.DataFrame, data_file: str, on_columns: List[str])->pd.DataFrame:

    if data_file == '**cell_shape.csv':

        column_name = 'cell_shape'

        df[column_name] = ""
        for i, row in df.iterrows():
            cell_shape_list = [row[column] for column in on_columns]
            df.at[i, column_name] = cell_shape_list

    else:

        column_name = 'protein_' + data_file.split('_')[-1][:-4]
        dapi_columns = [column for column in df.columns if 'DAPI' in column]

        df[column_name] = ""
        for i, row in df.iterrows():
            protein_dict = {column:row[column] for column in on_columns}
            df.at[i, column_name] = json.dumps(protein_dict)

        df.drop(dapi_columns, axis=1, inplace=True)

    df.drop(on_columns, axis=1, inplace=True)

    return df.copy()

def cluster_and_coalesce(df: pd.DataFrame, on_columns:List[str], data_file[str])->pd.DataFrame:

    cluster_column_header = 'cluster_by_' + data_file.split('_')[-1][:-4]

    data = df[on_columns].to_numpy()
    cluster_assignments = KMeans(n_clusters = 6, random_state = 0).fit(data)
    df[cluster_column_header] = pd.Series(cluster_assignments.labels_)
    return coalesce_columns(df, data_file, on_columns)

def stitch_dfs(data_file: str, dataset_directory: Path)->DataFrame:

    csv_files = list(get_csv_files(data_file, directory))

    tile_dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    tile_ids = [get_tile_id(csv_file) for csv_file in csv_files]

    for i, tile_df in enumerate(tile_dfs):
        tile_df['ID'] = tile_df['ID'].astype(str)
        tile_df['tile'] = tile_ids[i]
        tile_df['cell_id'] = modality + "-" + dataset + "-" + tile_df['tile'] + "-"+ tile_df['ID']
        if data_file == '**cell_cluster.csv':
            for column in tile_df.columns:
                if column != 'ID' and column != 'tile' and column != 'cell_id':
                    tile_df[column] = modality + "-" + dataset + "-" + tile_df['tile'] + "-"+ tile_df[column].astype(str)
                    print(tile_df[column].unique())
        tile_df.drop(['ID', 'tile'], axis=1, inplace=True)


    for tile_df in tile_dfs[1:]:
        tile_dfs[0] = tile_dfs[0].merge(tile_df, how='outer')

    if data_file != '**cell_shape.csv':
        protein_columns = [column for column in tile_dfs[0].columns if column != 'cell_id' and 'DAPI' not in column]
        tile_dfs[0] = cluster_and_coalesce(tile_dfs[0], protein_columns, data_file)

    return tile_dfs[0]

def main(nexus_token:str, output_directories:List[Path]):

    modality = 'codex'
    dataset_dfs = []
    database_file = Path('codex.db')

    per_cell_data_files = ['**cell_shape.csv', '**cell_channel_covar.csv', '**cell_channel_mean.csv', '**cell_channel_total.csv']

    for dataset_directory in output_directories:

        dataset = get_dataset(dataset_directory)
        tissue_type = get_tissue_type(dataset, nexus_token)

        stitched_dfs = [stitch_dfs(data_file, dataset_directory) for data_file in per_cell_data_files]

        dataset_df = stitched_dfs[0]

        for stitched_df in stitched_dfs[1:]:
            dataset_df = dataset_df.merge(stitched_df, how='outer')

        dataset_df['dataset'] = dataset
        dataset_df['tissue_type'] = tissue_type
        dataset_df['modality'] = modality

        dataset_dfs.append(dataset_df)

    modality_df = pd.concat(dataset_dfs)
    on_columns = [column for column in modality_df.columns if column.isdigit()]
    modality_df = cluster_and_coalesce(modality_df, on_columns, '**cell_shape.csv')

    modality_df.to_csv('codex.csv')


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('nexus_token', type=str)
    p.add_argument('output_directory', type=Path, nargs='+')
    args = p.parse_args()

    main(args.output_directory)
