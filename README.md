# Learning Theory for Kernel Bilevel Optimization
This repository contains the code to reproduce the experiments of the paper: **Learning Theory for Kernel Bilevel Optimization** (NeurIPS 2025) ([Paper Link](https://arxiv.org/abs/2502.08457))

To install the required dependencies, use the following command:
```bash
pip install -r requirements.txt
```

## Reproducing Experiments

Before running the first two commands below, configure the distribution in `config.py` by setting `dist="gaussian"` or `dist="student"` (and specifying $\nu$ for Student distribution). Repeat the following two commands for each distribution configuration.

1. First, run the following command to compute the blocks, replacing `/path/to/block` with the path to the directory where the blocks will be saved:
```bash
python generate_blocks.py --dir /path/to/block
```

2. Then, run the following command to compute the trajectories, replacing `/path/to/block` with the path to the directory containing the blocks and `/path/to/traj` with the path to the directory where the trajectories will be saved:

```bash
python main.py --block_dir /path/to/block --traj_dir /path/to/traj
```

Finally, run the following command to obtain generalization plots, replacing the paths with your actual trajectory directories for Gaussian, Student ($\nu=2.1$), Student ($\nu=2.5$), and Student ($\nu=2.9$) distributions respectively:

```bash
python plots.py --dirs /path/to/gaussian /path/to/student_df2.1 /path/to/student_df2.5 /path/to/student_df2.9
```

## Citation

If you find this work useful, please consider citing our paper:
```bibtex
@inproceedings{elkhoury2025kbo,
  title = {Learning Theory for Kernel Bilevel Optimization},
  author = {El Khoury, Fares and Pauwels, Edouard and Vaiter, Samuel and Arbel, Michael},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2025},
}
```
