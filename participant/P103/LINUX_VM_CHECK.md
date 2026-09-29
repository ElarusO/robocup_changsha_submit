# P103 Linux 虚拟机检验记录（2026-09-29）

## 当前已准备

- 主机 VMware Workstation 16.2.3；已有的 `Ubuntu 64 位` 虚拟机已启动，配置为 4 vCPU、4 GB 内存、NAT 网络。VMware Tools 状态为 `installed`。来宾终端确认系统为 Ubuntu 20.04.6 LTS，`x86_64`，根分区还有 8.7 GB，独立安装的 Python 为 3.12.13。
- 来宾未开放 SSH 22 端口；用户直接在虚拟机终端执行了安装、预检、评测和结果回传命令。无需把账户密码发到聊天中。
- VMware 共享目录 `P103Linux` 只读，包含 `robocup-linux-p103.zip`；该包来自当前本地工作树，包含官方评测代码、配置、依赖锁与 `participant/P103`，不含 Windows 虚拟环境、训练输出或 `.git`。压缩包 SHA-256 为 `d647cb7ea2b27f344193fbe1f9e4c425e62fac1bdd38e5fb29349f481a54329d`，Windows 路径为 `outputs/P103/linux-transfer-001/robocup-linux-p103.zip`。
- VMware 共享目录 `P103Results` 仅指向 `outputs/P103/linux-transfer-001/results/`，可把 Linux 结果复制回 Windows。两项共享都只涉及此次检验包目录；原仓库文件没有直接挂载给来宾写入。
- 包内 `participant/P103/linux_eval_setup.sh` 只在检测到 `python3.12` 后建立独立 `.venv-linux-p103`，安装官方锁定依赖，运行 E009 正式提交包的预检、公开套件、开发 v2/v3；每次输出使用时间戳新目录，不覆盖历史结果。脚本不自动执行 `sudo` 或改系统 Python。

## 在 Ubuntu 虚拟机终端先确认

```bash
cat /etc/os-release | head -n 3
python3.12 --version
ls /mnt/hgfs
df -h "$HOME"
```

若 `/mnt/hgfs` 不存在，但 `vmhgfs-fuse` 已安装，可在虚拟机中挂载 VMware 共享目录：

```bash
sudo mkdir -p /mnt/hgfs
sudo vmhgfs-fuse .host:/ /mnt/hgfs -o allow_other
ls /mnt/hgfs
```

### Ubuntu 20.04 安装独立的 Python 3.12

实测来宾为 `x86_64`，根分区还有 8.7 GB，尚无 Conda。Miniforge 官方 GitHub 下载在此虚拟机中发生 SSL 连接失败，留下的安装包校验失败，不能运行。当前 deadsnakes 的 Python 3.12 包目录未显示 Ubuntu 20.04/focal 构建。改用 [Anaconda 官方 Miniconda 归档](https://repo.anaconda.com/miniconda/)中已固定版本的 Python 3.12 安装包，不替换 Ubuntu 自带的 `python3`。2026-09-29 来宾终端报告安装包完整下载 161,366,816 字节，SHA-256 校验返回“成功”；Miniconda 已装至 `/home/user/miniconda3`，`python3.12 --version` 为 3.12.13。安装器发现来宾设置了 `PYTHONPATH`，运行项目前在当前终端执行了 `unset PYTHONPATH`。随后依赖安装、预检和三套评测全部完成。

```bash
cd "$HOME"
wget -O Miniconda3-py312_26.3.2-2-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-py312_26.3.2-2-Linux-x86_64.sh
echo '32673413a39a21ae3997c9b38236e9df15c9fcef930b510487c64fe259e03f95  Miniconda3-py312_26.3.2-2-Linux-x86_64.sh' | sha256sum -c -
bash Miniconda3-py312_26.3.2-2-Linux-x86_64.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/bin/activate"
python3.12 --version
```

若下载或 SHA-256 校验失败，停止安装并记录原始错误。每次新开终端都需重新执行 `source "$HOME/miniconda3/bin/activate"`，再运行下方脚本。此处 Miniconda 只用来提供 3.12 解释器；评测脚本仍会另建项目专用 `.venv-linux-p103` 并按照仓库锁文件安装依赖。

## 从只读共享目录复制代码，再运行

不要直接在 VMware 共享目录里安装 `.venv`。先复制并解压到虚拟机自己的磁盘：

```bash
unset PYTHONPATH
echo 'd647cb7ea2b27f344193fbe1f9e4c425e62fac1bdd38e5fb29349f481a54329d  /mnt/hgfs/P103Linux/robocup-linux-p103.zip' | sha256sum -c -
mkdir -p "$HOME/robocup-linux-check"
cd "$HOME/robocup-linux-check"
python3 -m zipfile -e /mnt/hgfs/P103Linux/robocup-linux-p103.zip .
bash participant/P103/linux_eval_setup.sh
```

脚本会输出预检状态、每套题的 `status`、`provenance`、回合数和分数。Linux 上若显示 `official_container`，只说明使用了带阶段超时的代码路径，不表示已经获得组织方正式核验。

## 把原始结果复制回 Windows

本次在解压后的仓库根目录执行了：

```bash
dest="/mnt/hgfs/P103Results/transfer-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$dest"
cp -r outputs/P103/linux-* "$dest/"
```

原始结果已回传至 `../../outputs/P103/linux-transfer-001/results/transfer-20260929-100241/`，含预检 `audit-report.json`、环境文本以及各套件的 `result.json`、`episodes.csv`、`timings.npz` 等文件。正式成绩仍以组织方核验为准。

## Windows 与 Linux 对照（同一 E009 模型）

以下 Windows `local_preview` 和 Linux 本地检验均使用模型 SHA-256 `6b5c8cc0fc0279a86e3218665e4623eeb10dd4a4b5e75c642933cb87fef6ed51`。两侧套件摘要、种子安排、推理依赖摘要一致；Linux 预检无硬拒绝，三套结果均为 `status=ok`、`errors=[]`。公开套件只有 8 回合，不能单凭该分数判断泛化能力。

| 套件 | 回合数 | Windows 分数 | Linux 分数 | Windows 结果目录 |
| --- | ---: | ---: | ---: | --- |
| public v1 | 8 | 200.0 | 200.0 | `outputs/P103/eval-E009` |
| validation v2 | 200 | 125.33333333333334 | 125.33333333333334 | `outputs/P103/validation-v2-E009-001` |
| validation v3 | 200 | 119.66666666666669 | 119.66666666666669 | `outputs/P103/e2-v3-E009-001` |

按 `group_id`、`case_id`、`repeat_index`、`attempt_index` 配对检查全部 408 个 `episodes.csv` 回合；忽略运行 ID 和耗时字段后，完成状态、覆盖、碰撞、回报及其他结果字段均相同，数值比较容差为 `1e-12`。两组的 `mean_j` 也逐套一致。Linux 结果中的 `provenance=official_container` 是评测器在 Linux 上启用超时路径的标签；`hardware_profile` 仍为 `host-local`、`official_lock_hash=local-dev-lock`、`worker_image_digest=null`，并非组织方容器镜像或隐藏测试。其 `memory_mib=8192` 是本地配置字段，不能当作虚拟机实际内存测量值（虚拟机配置为 4 GB）。此次只验证**当前 E009 提交包**；若后续换成 E015 等候选，应对最终包重新执行相同检验。
