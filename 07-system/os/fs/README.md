# File System

文件系统的核心作用：把磁盘、SSD 等块设备上的原始 block，组织成用户能理解的“目录 + 文件”模型，并提供命名、权限、缓存、空间分配、一致性恢复等能力。

从用户态看：

```text
path -> file descriptor -> read/write/mmap -> close
```

从内核态看：

```text
path lookup -> VFS -> inode/dentry/file -> page cache -> block layer -> device
```

## 1. 基本对象

### inode

inode 描述一个文件本身的元数据，不保存文件名。

常见字段：

- 文件类型：普通文件、目录、符号链接、设备文件等
- 权限：`rwx`
- 所属用户和用户组：`uid/gid`
- 文件大小
- 时间戳：`atime/mtime/ctime`
- link count：硬链接数量
- 数据块位置：直接或间接指向磁盘 block

一个文件可以有多个名字，但它们可以指向同一个 inode。

```bash
stat file
ls -li file
```

### dentry

dentry 是目录项缓存，表示“名字到 inode 的映射”。

例如：

```text
/home/star/a.txt
```

路径查找会逐级解析：

```text 
/ -> home -> star -> a.txt
```

每一级名字解析都依赖 dentry。Linux 会缓存 dentry，避免每次路径查找都访问磁盘。

### file

`file` 是一次 `open()` 产生的内核对象，描述“一个进程打开的文件实例”。

它保存：

- 当前读写偏移量
- 打开方式：只读、只写、追加、非阻塞等
- 指向 inode 的引用

文件描述符 fd 是进程 fd table 里的整数索引，指向内核中的 `file` 对象。

```text
process fd table
  fd=3 -> struct file -> inode -> data blocks
```

## 2. VFS

VFS，Virtual File System，是 Linux 对不同文件系统的统一抽象层。

用户程序调用：

```c
open("a.txt", O_RDONLY);
read(fd, buf, len);
write(fd, buf, len);
close(fd);
```

不需要关心底层是 ext4、xfs、btrfs、tmpfs、nfs，还是 procfs。VFS 会把统一的系统调用分发给具体文件系统实现。

典型抽象对象：

- superblock：描述一个挂载的文件系统实例
- inode：描述一个文件
- dentry：描述路径名解析缓存
- file：描述一次打开文件的上下文

## 3. 目录和普通文件

目录本质上也是一种文件，只是它的数据内容是“文件名到 inode 的映射”。

例如目录里有：

```text
name="a.txt" -> inode=1001
name="b.txt" -> inode=1002
```

所以：

- 文件名存在目录中
- inode 不保存自己的文件名
- rename 通常只是改目录项，不一定移动文件数据
- 删除文件通常是 unlink 一个名字，不一定立刻删除 inode 和数据

## 4. 硬链接和软链接

### 硬链接

硬链接是多个目录项指向同一个 inode。

```bash
echo hello > a.txt
ln a.txt b.txt
ls -li a.txt b.txt
```

特点：

- `a.txt` 和 `b.txt` inode 相同
- 修改任意一个，另一个看到同样内容
- `rm a.txt` 只是减少 link count
- link count 变成 0，并且没有进程打开它时，inode 和数据才会被释放

### 软链接

软链接，也叫符号链接，是一个特殊文件，内容是另一个路径。

```bash
ln -s a.txt s.txt
```

特点：

- 软链接有自己的 inode
- 内容是目标路径字符串
- 目标文件删除后，软链接会悬空
- 可以跨文件系统

## 5. open/read/write 的过程

### open

`open(path, flags)` 大致流程：

```text
1. 路径解析：根据 path 找到 dentry/inode
2. 权限检查：是否允许读写执行
3. 创建 struct file
4. 在进程 fd table 中分配 fd
5. 返回 fd 给用户态
```

### read

`read(fd, buf, len)` 大致流程：

```text
1. fd -> struct file
2. 根据 file offset 找到目标页
3. 优先从 page cache 读取
4. cache miss 时，从块设备读取到 page cache
5. 拷贝到用户态 buf
6. 更新 file offset
```

### write

`write(fd, buf, len)` 大致流程：

```text
1. fd -> struct file
2. 用户态数据拷贝到内核 page cache
3. 标记 page 为 dirty
4. write 可以先返回
5. 后台 writeback 再把 dirty page 写入磁盘
```

所以普通 `write()` 返回成功，不代表数据已经真正落盘。

如果需要更强的落盘语义：

```c
fsync(fd);      // 同步文件数据和必要元数据
fdatasync(fd); // 主要同步文件数据
```

## 6. Page Cache

page cache 是内核用内存缓存文件数据的机制。

它让文件 IO 不必每次都访问磁盘：

```text
read:  disk -> page cache -> user buffer
write: user buffer -> page cache -> later writeback -> disk
```

好处：

- 加速重复读取
- 合并小写入
- 利用空闲内存提升 IO 性能

需要注意：

- page cache 占用的内存不是“泄漏”，内核可以在内存压力下回收
- `free` 看到 buff/cache 很大是正常现象
- `O_DIRECT` 可以绕过 page cache，但对对齐、IO 大小等要求更严格

## 7. mmap

`mmap()` 可以把文件映射到进程虚拟地址空间。

```text
file -> page cache -> process virtual memory
```

访问映射地址时，如果页不在内存，会触发缺页异常，内核再把文件页加载进 page cache。

普通 `read/write` 和 `mmap` 都可能经过 page cache，只是访问方式不同：

- `read/write`：显式系统调用，用户 buffer 和内核 page cache 之间拷贝
- `mmap`：像访问内存一样访问文件页，减少一次用户态拷贝

## 8. 权限模型

Linux 文件权限由三组 `rwx` 组成：

```text
owner group others
rwx   rwx   rwx
```

例如：

```bash
chmod 644 a.txt
```

表示：

```text
owner:  read + write
group:  read
others: read
```

目录权限含义稍有不同：

- `r`：能列出目录项
- `w`：能在目录中创建、删除、重命名文件
- `x`：能进入目录，能进行路径穿越

因此目录没有 `x` 权限时，即使知道文件名，也可能打不开里面的文件。

## 9. 挂载

挂载是把一个文件系统实例接到目录树的某个挂载点。

```bash
mount /dev/sdb1 /data
```

挂载后，访问 `/data` 就是在访问 `/dev/sdb1` 上的文件系统。

Linux 的目录树是统一的，不像 Windows 用 `C:`、`D:` 区分盘符。不同设备、不同文件系统都可以挂载到同一棵树上。

常见伪文件系统：

- `procfs`：`/proc`，暴露进程和内核状态
- `sysfs`：`/sys`，暴露设备和驱动信息
- `tmpfs`：基于内存的临时文件系统
- `devtmpfs`：`/dev`，设备节点

## 10. 一致性和日志

文件系统需要处理突然断电、内核崩溃等情况。

如果写入过程中崩溃，可能出现：

- 数据块已经写入，但元数据没写
- 元数据已经写入，但数据块没写完
- 目录项、inode、bitmap 状态不一致

日志文件系统会先记录操作意图，再真正修改文件系统结构。

典型流程：

```text
1. 写 journal
2. journal commit
3. 修改真实 inode/block bitmap/data
4. 崩溃后根据 journal replay 或 rollback
```

ext4、xfs 都具备日志能力，但具体策略不同。

## 11. 常见系统调用

```c
open();
close();
read();
write();
pread();
pwrite();
lseek();
stat();
fstat();
fsync();
fdatasync();
rename();
unlink();
mkdir();
rmdir();
link();
symlink();
readlink();
mmap();
munmap();
```

几个容易混淆的点：

- `unlink()` 删除的是目录项，不是直接删除磁盘数据。
- `rename()` 在同一文件系统内通常具有原子性。
- `pread/pwrite` 不改变 `file` 对象里的当前 offset。
- `O_APPEND` 会让每次写入都追加到文件末尾。
- `fsync` 用于要求数据真正落盘，常见于数据库、日志系统。

## 12. 文件系统和块设备

磁盘/SSD 通常以 block 为单位读写。文件系统负责把文件偏移映射到设备 block。

```text
file offset -> logical block -> physical block -> device IO
```

空间管理常见结构：

- bitmap：记录 block 是否空闲
- extent：用一段连续区间描述文件数据位置
- inode table：保存 inode
- journal：保存一致性日志

现代文件系统一般更偏好 extent，因为连续空间描述更紧凑，也有利于顺序 IO。

## 13. 小结

一句话串起来：

```text
文件名在目录项里，元数据在 inode 里，打开状态在 file 对象里，数据常先进入 page cache，再由文件系统和块层写入设备。
```

理解文件系统时可以抓住这几条主线：

- 命名：路径、目录、dentry
- 元数据：inode、权限、时间戳、link count
- 打开状态：fd、file、offset、flags
- 缓存：page cache、dirty page、writeback
- 持久化：fsync、journal、崩溃恢复
- 设备映射：file offset 到 block 的映射
