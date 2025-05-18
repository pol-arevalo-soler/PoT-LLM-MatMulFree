# PoT MatMulFree LLM

## 1. Overview

This repository provides a collection of efficient Triton-based implementations for state-of-the-art flash-linear attention models, as in [flash-linear-attention](https://github.com/fla-org/flash-linear-attention). It extends these with a specialized PoT (Powers-of-Two) implementation.

## 2. PoT Implementation

The special implementation of PoT (Powers-of-Two quantization) can be found in:

```
src/matmulfreellm/mmfreelm/quantization.py
```

This module contains the core logic for quantizing weights using powers-of-two, enabling matrix multiplication-free inference in LLMs.

## 3. Code Base

This project is based on the following repository:

* [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
