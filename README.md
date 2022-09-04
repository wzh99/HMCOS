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

Zihan Wang, Chengcheng Wan, Yuting Chen, Ziyi Lin, He Jiang, and Lei Qiao. 2022. Hierarchical Memory-Constrained Operator Scheduling of Neural Architecture Search Networks. In Proceedings of the 59th ACM/IEEE Design Automation Conference (DAC) (DAC ’22), July 10–14, 2022, San Francisco, CA, USA. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3489517.3530472
