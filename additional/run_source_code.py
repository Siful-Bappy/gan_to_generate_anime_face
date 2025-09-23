"""

######################################################################################################

I found similar type of code in the google: 
1. https://www.kaggle.com/code/songseungwon/pytorch-gan-basic-tutorial-for-beginner
2. https://ddongwon.tistory.com/124
3. https://medium.com/codex/building-a-vanilla-gan-with-pytorch-ffdf26275b70

dataset_link: https://www.kaggle.com/datasets/splcher/animefacedataset

Dataset directory in my local machine: 
gan_to_generate_anime_face/dataset/anime_face_dataset/
                                            ├── images/
                                                    ├── 0_2000.jpg
                                                    ├── 1_2000.jpg
                                                    └── ...
                                            

######################################################################################################

run script:

# train.py
python train.py

# evaluation_chatgpt.py
python evaluation_chatgpt.py --checkpoint checkpoints/cnn_fmnist.pth --batch_size 128

# pix2Pix_gan.py
python pix2pix_gan.py \
  --data_root ./facades/train \
  --mode aligned \
  --epochs 100 \
  --batch_size 8

python pix2pix_gan.py \
  --data_root ./my_pairs \
  --mode folders \
  --folderA A --folderB B \
  --epochs 100 --batch_size 8


######################################################################################################

"""