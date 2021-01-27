#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.0
label: Pipeline for parsing and aggregating SPRM output across codex datasets

inputs:
  data_directory:
    label: "Path to processed codex dataset"
    type: Directory

  uuid:
    label: "32 character UUID corresponding to dataset"
    type: string

  nexus_token:
    label: "Valid nexus token for search-api"
    type: string

outputs:
  hdf5_file:
    outputSource: annotate-concatenate/hdf5_file
    type: File

  csv_file:
    outputSource: annotate-concatenate/csv_file
    type: File

steps:

  - id: annotate-concatenate
    in:
      - id: data_directory
        source: data_directory
      - id: nexus_token
        source: nexus_token
      - id: uuid
        source: uuid

    out:
      - hdf5_file
      - csv_file

    run: steps/annotate-concatenate.cwl
    label: "Annotates and concatenates csv files, writes out csvs"
