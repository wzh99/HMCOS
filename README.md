# HMCOS

This repository contains the implementation of our DAC'22 paper: Hierarchical Memory-Constrained Operator Scheduling of Neural Architecture Search Networks. 

## Introduction

Neural Architecture Search (NAS) is widely used in industry, searching for neural networks meeting task requirements. Meanwhile, it faces a challenge in scheduling networks satisfying memory constraints. We propose HMCOS that performs hierarchical memory-constrained operator scheduling of NAS networks: given a network, HMCOS constructs a hierarchical computation graph and employs an iterative scheduling algorithm to progressively reduce peak memory footprints. 

## Dependency

To run HMCOS, a C++17-compatible compiler and the following C++ libraries are required:

* [ONNX](https://github.com/onnx/onnx) 1.9
* [Protobuf](https://github.com/protocolbuffers/protobuf) 3.11
* [glog](https://github.com/google/glog)
* [fmt](https://github.com/fmtlib/fmt)

To support graph visualization features in HMCOS, [Graphviz](https://graphviz.org/) is also required.

To generate ONNX models with [Python scripts](script), some Python packages are required. Run `pip install -r requirements.txt` to install them. 

## Usage

### Executable

Compile target `op_sched` and run `./op_sched ${modelPath} ${outputDir}`.

### Source

Check [op_sched.cpp](src/bin/op_sched.cpp) for sample usage of HMCOS API. 

## Citation

```bibtex
@inproceedings{wang2022hierarchical,
    author = {Wang, Zihan and Wan, Chengcheng and Chen, Yuting and Lin, Ziyi and Jiang, He and Qiao, Lei},
    title = {Hierarchical Memory-Constrained Operator Scheduling of Neural Architecture Search Networks},
    year = {2022},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3489517.3530472},
    doi = {10.1145/3489517.3530472},
    booktitle = {Proceedings of the 59th ACM/IEEE Design Automation Conference},
    pages = {493–498},
    numpages = {6},
    location = {San Francisco, California},
    series = {DAC '22}
}
```