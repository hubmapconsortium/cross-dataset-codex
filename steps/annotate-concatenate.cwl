cwlVersion: v1.0
class: CommandLineTool
label: Annotates each h5ad file with dataset and tissue type, then concatenates

hints:
  DockerRequirement:
    dockerPull: hubmap/cross-dataset-sprm:latest
baseCommand: /opt/parse_sprm_output.py

inputs:

  nexus_token:
    type: string
    doc: Valid nexus token for search-api
    inputBinding:
      position: 1

  data_directories:
    type: string[]
    doc: List of paths to processed dataset directories
    inputBinding:
      position: 2

outputs:
  csv_files:
    type: File[]
    doc: csv files containing annotated and concatenated cell and protein data
    outputBinding:
      glob: "*.csv"
