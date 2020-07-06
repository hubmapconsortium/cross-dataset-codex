#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.0
label: Pipeline for parsing and aggregating SPRM output across codex datasets

inputs:
  data_dir:
    label: "Directory containing h5ad data files"
    type: Directory

outputs:
  db_file:
    outputSource: join-annotate/db_file
    type: File

steps:
  - id: join-annotate
    in:
      - id: data_dir
        source: data_dir

    out:
      - db_file

    run: steps/join-annotate.cwl
    label: "Annotates and concatenates csv files, writes out relational db"
