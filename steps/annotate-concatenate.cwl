cwlVersion: v1.0
class: CommandLineTool
label: Annotates each h5ad file with dataset and tissue type, then concatenates

hints:
  DockerRequirement:
    dockerPull: hubmap/cross-dataset-codex:single

  cwltool:LoadListingRequirement:
    loadListing: shallow_listing
baseCommand: /opt/parse_sprm_output.py

inputs:

  nexus_token:
    type: string
    doc: Valid nexus token for search-api
    inputBinding:
      position: 1

  data_directory:
    type: Directory
    doc: Path to dataset directory
    inputBinding:
      position: 2

  uuid:
    type: string
    doc: String representation of 32 character UUID
    inputBinding:
        position: 3

outputs:
  hdf5_file:
    type: File
    doc: hdf5 file containing dataframes for cell and group level data
    outputBinding:
      glob: "codex.hdf5"
