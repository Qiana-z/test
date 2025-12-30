### Always Step 1：创建虚拟环境
```
python3 -m venv <venv_name>  # 创建虚拟环境
source <venv_name>/bin/activate  # 激活
pip install -r requirements.txt # 会把依赖安装在venv里
```

### 本地文件上传到gitlab/codebase等代码仓库
#### Step 1: 进入本地文件夹
```
cd model_beg                  # 进入你的文件夹
```

#### Step 2: 创建.gitignore文件把不需要的ignore了，比如下列文件一般不传入git
```
/venv
/model
site-packages/
__pycache__/
```

#### Step 3: 连接远程git仓库
```
git init --initial-branch=main  # 初始化一个Git仓库，并指定主分支叫main
git remote add origin git@code.byted.org:qianhui.zhang/model_train.git  # 添加远程仓库

git add .                     # 把当前所有文件加入暂存区
git commit -m "Initial commit" # 提交一次初始提交

git push -u origin main       # 推送到远程origin的main分支，并建立追踪关系
```

### Merlin开发机指令
```
mlx worker quota    # 查看当前空闲资源
launch --type=a100-80g -- bash    # 拉起gpu worker
source vmodel/bin/activate    # 关联开发机本地虚拟环境
source /root/miniconda3/etc/profile.d/conda.sh    # 或者用conda环境
conda activate qwen3vl
exit    # 退出gpu worker
```

```
# 查看当前开发机是否有gpu支持
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

### conda 指令
```
conda list -n <env_name> python # 查看当前 conda 环境使用的python版本
conda env list 

# 把当前运行命令行path切换到conda里
export PATH=/root/miniconda3/envs/qwen3vl/bin:$PATH
hash -r

# 验证
which python3
python3 -c "import sys; print(sys.executable)"
# 会看到
/root/miniconda3/envs/qwen3vl/bin/python3
```

### hf下载相关
```
# 1. 安装依赖
pip install -U huggingface_hub

# 2. 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 3. 使用 huggingface-cli 下载模型
huggingface-cli download \
--resume-download Qwen/Qwen3-Omni-30B-A3B-Instruct \  # Hugging Face 上的模型仓库名
--local-dir model/Qwen/Qwen3-Omni-30B-A3B-Instruct  # 本地指定目录

# 4. 使用 huggingface-cli 下载数据集
huggingface-cli download \
--repo-type dataset \
--resume-download wikitext \  # Hugging Face 上的数据集名
--local-dir wikitext  # 本地指定目录
```

### 其他linux指令
#### 0. 未归档
```
pip show <package_name>  # 查看包版本
pip install --upgrade <package_name>   #升级包版本     
```

#### 1. 通过 git lfs 上传大型model文件至远程git仓库
```
# 下载lfs工具
git lfs install
git lfs version

# 生成.gitattributes文件，记得这个文件要在根目录
git lfs track "*.bin"
git lfs track "*.safetensors"
```

#### 2. 仅提取当前文件中 import 的包
```
pip install pipreqs  
pipreqs . --force
```

#### 3. gpu相关
```
nvidia-smi  # 作用：查看这台机器是否真的有 NVIDIA GPU以及相关memory usage
```