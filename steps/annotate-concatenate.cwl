cwlVersion: v1.0
class: CommandLineTool
label: Annotates each h5ad file with dataset and tissue type, then concatenates

hints:
  DockerRequirement:
    dockerPull: hubmap/cross-dataset-codex:latest
baseCommand: /opt/parse_sprm_output.py

inputs:

  nexus_token:
    type: string
    doc: Valid nexus token for search-api
    inputBinding:
      position: 1

  data_directories:
    type: Directory[]
    doc: List of paths to processed dataset directories
    inputBinding:
      position: 2

outputs:
  hdf5_file:
    type: File
    doc: hdf5 file containing dataframes for cell and group level data
    outputBinding:
      glob: "codex.hdf5"
