# 下载所有数据集

```bash
cd /root/SoulX-Duplug/datasets
export HF_TOKEN=hf_xxx
export WENETSPEECH_PASSWORD='你的 WenetSpeech 官方密码'
./download_all_datasets.sh
```

也可以用密码文件：

```bash
export WENETSPEECH_PASSWORD_FILE=/path/to/wenetspeech_password.txt
```

重复执行不会重复创建虚拟环境；默认复用：

```bash
/root/SoulX-Duplug/datasets/.dataset-download-env
```

如果下载结束后要删除虚拟环境：

```bash
DELETE_ENV_AFTER=1 ./download_all_datasets.sh
```

日志文件：

```bash
/root/SoulX-Duplug/datasets/download.log
```

```bash
tail -f /root/SoulX-Duplug/datasets/download.log
```

日志中重点看：

```text
[dataset_start]    开始处理某个数据集
[file_start]       开始下载某个文件及目标路径
[progress]         下载进度
[file_done]        文件下载完成及存储路径
[extract_done]     解压完成及目录
[dataset_success]  数据集完成及存放位置
[hf_list]          正在从 Hugging Face 获取文件列表
[hf_list_warn]     Hugging Face 文件列表请求失败并准备重试
[wenetspeech]      WenetSpeech 官方 toolkit 下载/解压输出
[error]            失败原因
```

注意：

```text
WenetSpeech 需要先在官网申请密码；没有密码无法自动下载。
WenetSpeech 官方 toolkit 会自动 clone、自动写入 SAFEBOX/password、自动调用下载脚本。
下载脚本结束后会删除自动 clone 的 toolkit 和 SAFEBOX/password，保持目录整洁。
CommonVoice 中文/英文使用 VoxBox 中的 commonvoice_cn/commonvoice_en 子集。
VoxBox/Emilia/CommonVoice 通过 Hugging Face 下载，网络不稳定时会自动重试。
```
