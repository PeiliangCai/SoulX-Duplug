# 下载所有数据集

```bash
cd /root/SoulX-Duplug/datasets
export HF_TOKEN=hf_xxx
./download_all_datasets.sh
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
[error]            失败原因
```
