# git clone minimal-cefr
```
git clone https://github.com/berstearns/minimal-cefr.git
cd minimal-cefr 
```

# setup enviroment 
## docker
```

```

## python with pyenv
```
apt-get update && apt-get install pyenv
pyenv install 3.10
```

find the python path ~/.pyenv/versions/3.10.16/bin/python
setup as default python
```
```

## gh 
```
apt-get install gh
gh auth login
```


## rclone for data
```
apt-get install rclone
```

```
gh repo clone pn
cp ~/pn/os_configs/rclone/rclone.conf ~/.config/rclone/rclone.conf
```

# get data
## setup folder structure 
```
rclone copy --progress --max-size 1m --transfers=3 --ignore-existing --create-empty-src-dirs i:/phd-experimental-data/cefr-classification/ ~/i/phd-experimental-data/cefr-classification/
```
## get a experiment folder
```
rclone copyto --max-size 100M --progress i:/phd-experimental-data/cefr-classification/data/experiments .
tar -xzf 90-10-init.tar.gz
```

# run pipeline 
