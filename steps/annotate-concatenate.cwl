cwlVersion: v1.0
class: CommandLineTool
label: Annotates each h5ad file with dataset and tissue type, then concatenates


hints:
  DockerRequirement:
    dockerPull: hubmap/cross-dataset-codex

baseCommand: /opt/parse_sprm_output.py

inputs:

  enable_manhole:
    label: "Whether to enable remote debugging via 'manhole'"
    type: boolean?
    inputBinding:
      position: 0

  nexus_token:
    type: string?
    doc: Valid nexus token for search-api
    inputBinding:
      position: 1
    default: "None"

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

  h5ad_file:
    type: File
    doc: h5ad file containing numeric expression data and cell annotations
    outputBinding:
      glob: "codex.h5ad"

  adjacency_file:
    type: File
    doc: h5ad file containing cell by cell distance/adjacency matrix
    outputBinding:
      glob: "codex_adjacency.h5ad"

  precompute_file:
    type: File
    doc: hdf5 file containing precomputed results for accelerated queries
    outputBinding:
      glob: "codex_precompute.hdf5"