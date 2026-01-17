

# Fair Multimodal Recommendation

This repository includes the implementation for paper *Causality-Inspired Fair Representation Learning for Multimodal Recommendation*.

## Datasets
The preprocessed MovieLens-1M dataset are already provided in the `./data/ml1m` folder. The proprocessed data of MicroLens dataset could be downloaded from [MicroLens-Fairness](https://recsys.westlake.edu.cn/MicroLens-Fairness-Dataset/).

## Environments

The experimental environment is Python 3.10.11. We can first create and activate a new [Anaconda](https://www.anaconda.com/) environment for Python 3.10.11:
```
> conda create -n FMMRec python=3.10.11
> conda activate FMMRec
```

Then install all the required packages by using the command:
```
> pip install -r ./requirements.txt
```

## Usage
The used disentangled modal embeddings are already contained in the `./data/[dataset]/` folder. To run the disentanglement learning, for example, we could run the following code for visual modality on the MicroLens dataset:
```
> python BMMF_runner.py --dataset microlens --modality v --gpu_id 0 --epochs 100
```

For the MovieLens dataset, we can run the code of the assembly of FMMRec fairness method on LATTICE recommendation model by running this command:
```
> cd ./src/
> nohup python -u main.py --fairness_model BFMMR --knn_k_uugraph 10 --filter_mode shared --prompt_mode concat --recommendation_model LATTICE --dataset ml1m --d_steps 10 --gpu_id 1 > MovieLens.out 2>&1 &
```

For the MicroLens dataset, we can run the code of the assembly of FMMRec fairness method on DRAGON recommendation model by running this command:
```
> cd ./src/
> nohup python -u main.py --fairness_model BFMMR --knn_k_uugraph 7 --filter_mode shared --prompt_mode concat --recommendation_model DRAGON --dataset microlens --d_steps 10 --gpu_id 0 > MicroLens.out 2>&1 &
```

## Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{chen2025causality,
  title     = {Causality-Inspired Fair Representation Learning for Multimodal Recommendation},
  author    = {Chen, Weixin and Chen, Li and Ni, Yongxin and Zhao, Yuhan},
  year      = 2025,
  journal   = {ACM Transactions on Information Systems},
  volume    = {43},
  number    = {6},
  articleno = {153},
  numpages  = {29}
}
```



## Acknowledgement
The code of this repository is implemented based on the multimodal recommendation framework at [MMRec](https://github.com/enoche/MMRec).

