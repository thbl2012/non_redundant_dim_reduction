# Non-redundant Dimensionality Reduction Methods and Applications in Image Compression

## Math 797 Spring 2020 Final Project

### Author: Linh Tran

**Abstract:** This report aims to give a summary and some critical comments of the [paper titled ``Non-redundant Spectral Dimensionality Reduction''](https://link.springer.com/chapter/10.1007/978-3-319-71249-9_16) by Yochai Blau and Tomer Michaeli.

The output of dimensionality reduction algorithms consists of components (equivalently, projections), each corresponding to one new dimension in the reduced data. A dimension is redundant if it is heavily correlated to one or many dimensions before it. In many cases, the correlation is so high that the redundant dimension approximates a deterministic function of previous ones. In particular, for two-dimensional embeddings, redundancy causes the shape of the output data to look narrow, more resemblant of a one-dimensional strip than the supposedly two-dimensional shape of the input data.
The authors of the aforementioned paper came up with significant modifications to the well-known non-linear dimensionality reduction methods, including Isomap, Locally Linear Embedding, and Local Tangent Space Alignment.

We reproduced an experiment in the paper and produced several new ones to verify the redundant dimension phenomenon discussed in the paper. In the same experiments, we also tested our own redundancy removal algorithm, based on the same mathematical foundations as the authors' algorithm.

### Detailed Report

For the detailed report, see the file [non_redundant_dim_reduction.pdf](non_redundant_dim_reduction.pdf).
