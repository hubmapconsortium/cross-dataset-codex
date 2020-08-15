import yaml
from pathlib import Path

def main(data_dir_log:Path, nexus_token:str):

    yaml_dict = {}

    with open(data_dir_log) as file:
        lines = [line.strip() for line in file]
        yaml_dict['data_directories'] = lines

    yaml_dict['nexus_token'] = nexus_token

    with open('data_directories.yml') as file:
        yaml.dump(yaml_dict, file)

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('data_dir_log', type=Path)
    p.add_argument('nexus_token', type=str)
    args = p.parse_args()

    main(args.data_dir_log, args.nexus_token)
