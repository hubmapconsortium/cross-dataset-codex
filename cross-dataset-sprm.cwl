#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.0
label: Pipeline for parsing and aggregating SPRM output across codex datasets

inputs:
  data_dir_log:
    label: "Text file containing paths to all processed RNA datasets"
    type: File

  nexus_token:
    label: "Valid nexus token for search-api"
    type: String

outputs:
  csv_files:
    outputSource: join-annotate/csv_files
    type: File[]

steps:

  - id: read-data-dir-log:
    in:
      - id: data_dir_log
        source: data_dir_log
    out:
      - data_directories

    run: steps/read-data-dir-log.cwl
    label: "Reads the log containing processed datasets"

  - id: join-annotate
    in:
      - id: data_directories
        source: read-data-dir-logs/data_directories
      - id: nexus_token
        source: nexus_token

    out:
      - csv_files

    run: steps/join-annotate.cwl
    label: "Annotates and concatenates csv files, writes out csvs"
