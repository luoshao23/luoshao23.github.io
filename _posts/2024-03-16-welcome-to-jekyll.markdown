---
layout: post
title:  "Useful command"
date:   2024-03-16 23:58:15 +0800
categories: miscellaneous
---

# Useful Commands


### Jar包反编译

<https://github.com/skylot/jadx/releases/tag/v1.4.7>

```python
./jadx 要反编译的jar包
```

### 挂载硬盘

```bash
fdisk -l
mount /dev/vdf /data4/
vim /etc/fstab # 增加配置后重启不会消失
# --- vim ---
/dev/vdf /data4  ext4  defaults 0 0
```

### nc传输文件

```bash
search-wk-793接收：nc -l -p 12346 > mysql_input.txt
search-test05发送：nc 10.0.213.163 12346 < mysql_input.txt
```

### 端口转发

```bash
socat -d -d -d -lf socat.log TCP4-LISTEN:9235,bind=10.0.50.164,reuseaddr,fork TCP4:10.11.178.5:6006
```

### yarn

<https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/ResourceManagerRest.html#Cluster_Application_Priority_API>

```bash
yarn application -list
#提高优先级
yarn application -updatePriority 40 -appId application_1636428463638_5247
curl -X PUT -H "Content-Type: application/json" 'http://ip-10-54-227-0.cn-north-1.compute.internal:8088/ws/v1/cluster/apps/application_1668742473322_241400/priority' -d '{"priority":41}'
curl -H "Content-Type: application/json" 'http://49.234.242.34:8088/ws/v1/cluster/apps'
```

### tmux

rename name of window: `C-b :rename-window <new name>`  or `Ctrl + b + ,`


### io 带宽

```bash
iostat -x 1
```

### es 中 前缀匹配配置

要加`keyword`，然后使用prefixQuery

> <https://www.elastic.co/guide/en/elasticsearch/reference/6.5/query-dsl-prefix-query.html>
>

```bash
"value": {
    "type": "text",
    "fields": {
      **"keyword": {
          "type": "keyword",
          "ignore_above": 256**
      }
    }
}
```

### conda env export & create

[Managing environments - conda 4.11.0.post12+f00535029 documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

```bash
# export
conda env export > environment.yml
# create
conda env create -f environment.yml
```

<https://zhuanlan.zhihu.com/p/87344422>

安装Nvidia版本的TF（python3.8）
<https://github.com/nvidia/tensorflow>

$ pip install --user nvidia-pyindex
$ pip install --user nvidia-tensorflow[horovod]

有很多包无法直接下载，根据提示手动下载对应版本whl，这个最花时间
<https://developer.download.nvidia.com/compute/redist/nvidia-curand-cu115/>

出现GLIBC 版本不对，要按照22.01 release以后的，libc 2.12以上就可以了。
Error /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found

<https://github.com/NVIDIA/tensorflow/issues/32>

出现protobuf版本不对，注意安装版本不要太高，比如按照3.14.0就可以

其他python 包下载地址
<https://mirrors.aliyun.com/pypi/simple/numpy/https://mirrors.cloud.tencent.com/pypi/simple/protobuf/>

环境打包

Pip install conda-pack

### **jupyer环境多个kernel配置**

conda install ipykernel

python -m ipykernel install --user --name gemma --display-name "Python (gemma)”


### rocksdb

<https://python-rocksdb.readthedocs.io/en/latest/installation.html>

```bash
# 1. copy rocksdb 头文件目标 编译好的动态库到

cp somewhere/librocksdb.so.6.14.6 /usr/lib64/librocksdb.so.6.14.6
ln -s /usr/lib64/librocksdb.so.6.14.6 /usr/lib64/librocksdb.so.6
ln -s /usr/lib64/librocksdb.so.6.14.6 /usr/lib64/librocksdb.so
cp /usr/include/rocksdb /usr/include/rocksdb
# 2. 安装依赖

yum install bzip2 bzip2-devel
yum install snappy snappy-devel
yum install lz4-devel

# 3. copy 到虚拟环境lib 路径(如/data/app/anaconda3/envs/tf_1.14)
mylib='/data/user/miniconda3/envs/tf_1.15/lib'
cd $mylib
cp /usr/lib64/liblz4.so.1.7.5  .
cp /usr/lib64/libsnappy.so.1.1.4  .
cp /usr/lib64/libbz2.so.1.0.6 .

# 5. 创建软链接
ln -s liblz4.so.1.7.5 liblz4.so
ln -s libsnappy.so.1.1.4 libsnappy.so
ln -s libbz2.so.1.0.6 libbz2.so

# 6. 安装python package
pip install python-rocksdb

# 7. test
import rocksdb
db = rocksdb.DB("test.db", rocksdb.Options(create_if_missing=True))
db.put(b"a", b"b")
db.get(b"a")
```

### 定时触发任务

```bash
* * * * flock -xno /tmp/vid_job.lock -c "/data/train/vid_code/cmds/vid_daily.sh > /data/train/daily_log/vid 2>&1”
```

### brew

`HOMEBREW_NO_AUTO_UPDATE=1 brew install`

<!-- You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-title.MARKUP`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `MARKUP` is the file extension representing the format used in the file. After that, include the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/ -->
