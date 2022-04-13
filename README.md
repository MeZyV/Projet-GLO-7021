# Projet GLO-7021 : _SuperPoint_
Code pour le projet GLO-7021

Reproduction de l'article : https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf

Implémentations :
 - https://github.com/rpautrat/SuperPoint
 - https://github.com/eric-yyjau/pytorch-superpoint/
 - https://github.com/omercohen93/superpoint-pytorch
 - https://github.com/magicleap/SuperPointPretrainedNetwork

Démo :
```bash
python ./magicleap_demo_superpoint.py
```

Generate synthetic_shapes :
```bash
python ./generate_synthetic_dataset.py
```

Train on synthetic_shape :
```bash
python ./train_synthetic_shape.py
```

Run TensorBoard server :
```bash
tensorboard serve --logdir ./logs
```

# TODO

 - [ ] Reproduire article
 - [ ] Ameliorer modele
 - [ ] Trouver des articles en liens
 - [ ] Kaggle