
# PAI-Conv: Permutable Anisotropic Convolutional Networks for Learning on Point Clouds [Classification]

![Pai-Conv](data/images/pai-conv.png "Pai-Conv operation")
## Point Cloud Classification
* Run the training script:


``` 1024 points
python main.py --exp_name=paigcnn_1024 --model=paigcnn --num_points=1024 --k=20 --use_sgd=True
```

``` 2048 points
python main.py --exp_name=paigcnn_2048 --model=paigcnn --num_points=2048 --k=40 --use_sgd=True
```

* Run the evaluation script after training finished:

``` 1024 points
python main.py --exp_name=paigcnn_1024_eval --model=paigcnn --num_points=1024 --k=20 --use_sgd=True --eval=True --model_path=checkpoints/paigcnn_1024/models/model.t7
```

``` 2048 points
python main.py --exp_name=paigcnn_2048_eval --model=paigcnn --num_points=2048 --k=40 --use_sgd=True --eval=True --model_path=checkpoints/paigcnn_2048/models/model.t7
```

* Run the evaluation script with pretrained models:

``` 1024 points
python main.py --exp_name=paigcnn_1024_eval --model=paigcnn --num_points=1024 --k=20 --use_sgd=True --eval=True --model_path=pretrained/model.1024.t7
```

``` 2048 points
python main.py --exp_name=paigcnn_2048_eval --model=paigcnn --num_points=2048 --k=40 --use_sgd=True --eval=True --model_path=pretrained/model.2048.t7
```
