#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.0
label: Pipeline for parsing and aggregating SPRM output across codex datasets

inputs:
  data_directories:
    label: "List of paths to all processed RNA datasets"
    type: Directory[]

  nexus_token:
    label: "Valid nexus token for search-api"
    type: string

outputs:
  csv_files:
    outputSource: annotate-concatenate/csv_files
    type: File[]

steps:

  - id: annotate-concatenate
    in:
      - id: data_directories
        source: data_directories
      - id: nexus_token
        source: nexus_token

    out:
      - csv_files

    run: steps/annotate-concatenate.cwl
    label: "Annotates and concatenates csv files, writes out csvs"
