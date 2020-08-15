cwlVersion: v1.0
class: CommandLineTool
label: Annotates each h5ad file with dataset and tissue type, then concatenates

hints:
  DockerRequirement:
    dockerPull: hubmap/cross-dataset-atac:latest
baseCommand: /opt/read_data_logs.py

inputs:
  data_dir_log:
    type: File
    doc: Text file containing list of directories of processed datasets
    inputBinding:
      position: 1

outputs:
  data_directories:
    type: File
    doc: yaml file containing data directories and nexus token, args for next step
    outputBinding:
      glob: 'data_directories.yml'
