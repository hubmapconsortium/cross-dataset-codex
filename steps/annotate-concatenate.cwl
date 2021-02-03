cwlVersion: v1.0
class: CommandLineTool
label: Annotates each h5ad file with dataset and tissue type, then concatenates


hints:
  DockerRequirement:
    dockerPull: hubmap/cross-dataset-codex

baseCommand: /opt/parse_sprm_output.py

inputs:

  nexus_token:
    type: string
    doc: Valid nexus token for search-api
    inputBinding:
      position: 1

  data_directories:
    type: Directory[]
    doc: Path to dataset directory
    inputBinding:
      position: 2


outputs:
  hdf5_file:
    type: File
    doc: hdf5 file containing dataframes for cell and group level data
    outputBinding:
      glob: "codex.hdf5"

  csv_file:
    type: File
    doc: csv file containing dataframes for quantitative data
    outputBinding:
      glob: "codex.csv"

  mini_hdf5_file:
    type: File
    doc: hdf5 file containing dataframes for cell and group level minimal data
    outputBinding:
      glob: "mini_codex.hdf5"

  mini_csv_file:
    type: File
    doc: csv file containing dataframes for minimal quantitative data
    outputBinding:
      glob: "mini_codex.csv"