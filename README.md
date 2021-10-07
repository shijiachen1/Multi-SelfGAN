# Multi-SelfGAN
### environment requirements:
python >= 3.6

torch >= 1.1.0 

```bash
pip install -r requirements.txt
```

### prepare fid statistic file
 ```bash
mkdir fid_stat
 ```
Download the pre-calculated statistics
([Google Drive](https://drive.google.com/drive/folders/1UUQVT2Zj-kW1c2FJOFIdGdlDHA3gFJJd?usp=sharing)) to `./fid_stat`.


## How to search & train the derived architecture by yourself
```bash
sh exps/autogan_search_preblock.sh
```

When the search algorithm is done, you will get a vector denoting the discovered architecture, which can be viewed in the "*.log" file. 

To train from scratch and get the performance of your discovered architecture, run the following command (you should replace the architecture vector following "--arch" with yours):

```bash
python train_derived.py \
-gen_bs 128 \
-dis_bs 64 \
--dataset cifar10 \
--bottom_width 4 \
--img_size 32 \
--max_iter 80000 \
--gen_model shared_gan \
--dis_model shared_gan \
--latent_dim 128 \
--gf_dim 256 \
--df_dim 128 \
--g_spectral_norm False \
--d_spectral_norm True \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 5 \
--val_freq 20 \
--arch 0 1 0 1 0 0 0 1 0 0 1 1 1 0 \
--exp_name derive
```

## Citation


## Acknowledgement
1. Inception Score code from [OpenAI's Improved GAN](https://github.com/openai/improved-gan/tree/master/inception_score) (official).
2. FID code and CIFAR-10 statistics file from [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR) (official).
3. Thanks codes from AutoGAN [https://github.com/TAMU-VITA/AutoGAN](https://github.com/TAMU-VITA/AutoGAN)

