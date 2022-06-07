#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.0
label: Pipeline for parsing and aggregating SPRM output across codex datasets

inputs:

  enable_manhole:
    label: "Whether to enable remote debugging via 'manhole'"
    type: boolean?

  data_directories:
    label: "Path to processed codex dataset"
    type: Directory[]

  nexus_token:
    label: "Valid nexus token for search-api"
    type: string?

outputs:
  hdf5_file:
    outputSource: annotate-concatenate/hdf5_file
    type: File

  h5ad_file:
    outputSource: annotate-concatenate/h5ad_file
    type: File

  adjacency_file:
    outputSource: annotate-concatenate/adjacency_file
    type: File

  precompute_file:
    outputSource: annotate-concatenate/precompute_file
    type: File

steps:

  - id: annotate-concatenate
    in:
      - id: enable_manhole
        source: enable_manhole
      - id: data_directories
        source: data_directories
      - id: nexus_token
        source: nexus_token

    out:
      - hdf5_file
      - h5ad_file
      - adjacency_file
      - precompute_file

    run: steps/annotate-concatenate.cwl
    label: "Annotates and concatenates csv files, writes out csvs"
