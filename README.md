# motor-system

![Misaki](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/3329d75f-6071-400c-a30e-ffb4a845f064/d6kq62f-f08949f6-f5fb-40d0-9847-fbbc35caa7ed.png/v1/fill/w_1600,h_900,strp/misaki_shokuhou___wallpaper_by_rankwinner_d6kq62f-fullview.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9OTAwIiwicGF0aCI6IlwvZlwvMzMyOWQ3NWYtNjA3MS00MDBjLWEzMGUtZmZiNGE4NDVmMDY0XC9kNmtxNjJmLWYwODk0OWY2LWY1ZmItNDBkMC05ODQ3LWZiYmMzNWNhYTdlZC5wbmciLCJ3aWR0aCI6Ijw9MTYwMCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.eC6HGnG8cK7DH_CbL8zwZ0RGsmxDgapWp0pD3eLfUCI) 

## Introduction
A code copied from google-research which named motion-imitation was rewrited with PyTorch.
More details can get from this project. 

GIthub Link:https://github.com/google-research/motion_imitation

Project Link:https://xbpeng.github.io/projects/Robotic_Imitation/index.html

## Tutorials
For training：
```
python motion_imitation/run_torch.py --mode train --motion_file 'dog_pace.txt|dog_spin.txt' /
--int_save_freq 10000000 --visualize --num_envs 50 --type_name 'dog_pace'
```
* mode: train or test
* motion_file: Chose which motion to imitate (ps: | is used to split different motion)
* visualize: Whether rendering or not when training
* num_envs: Number of environments calculated in parallel

For testing:
```
python motion_imitation/run_torch.py --mode test --motion_file 'dog_pace.txt' --model_file 'model_file_path' --visualize
```

* model_path: There's a model parameters zip file, you just find out and copy it's path.

## Extra work
In this project, I donot use Gaussian distribution to fitting the encoder rather by using a mlp network with one hidden layer. The loss function is z(output of net)*advantages. And now I am testing the performance.
