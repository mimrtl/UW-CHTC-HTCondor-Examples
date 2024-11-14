
# Guide to Commands for Remote Access, File Transfer, Job Submission, and File Compression

This guide explains each command necessary for accessing remote servers, transferring files, managing job submissions on a high-throughput computing system, and compressing/decompressing files.

## Table of Contents
1. [Remote Access](#remote-access)
2. [File Transfer](#file-transfer)
3. [Job Management with HTCondor](#job-management-with-htcondor)
4. [File Compression and Decompression](#file-compression-and-decompression)
5. [Group GitHub Page](#group-github-page)

---

### 1. Remote Access

To connect to the high-throughput computing server (HTCondor), use SSH:
```bash
ssh yourWiscID@ap2001.chtc.wisc.edu
```
- `ssh`: Securely connects to a remote server.
- `yourWiscID`: Replace this with your WiscNet ID.
- `ap2001.chtc.wisc.edu`: The server’s hostname where HTCondor is running.

### 2. File Transfer

To transfer files from your local machine to the HTCondor server, use SCP:
```bash
scp your_file_name yourWiscID@ap2001.chtc.wisc.edu:~/your_folder
```
- `scp`: Securely copies files between your local machine and a remote server.
- `your_file_name`: The name of the file you want to transfer.
- `yourWiscID@ap2001.chtc.wisc.edu`: The server’s address.
- `~/your_folder`: The destination folder on the server where the file will be stored.

### 3. Job Management with HTCondor

HTCondor commands help manage job submissions and monitor their status.

#### Submit a Job
```bash
condor_submit try_wandb.sub
```
- `condor_submit`: Submits a job description file (`try_wandb.sub`) to the HTCondor scheduler for processing.

#### View Job Queue
```bash
condor_q
```
- `condor_q`: Lists all the jobs you have submitted to the HTCondor queue, displaying their current status (e.g., idle, running).

#### Monitor Job Queue in Real-Time
```bash
condor_watch_q
```
- `condor_watch_q`: Continuously monitors the HTCondor job queue, updating job statuses in real-time.

#### Remove a Job
```bash
condor_rm
```
- `condor_rm`: Removes (cancels) a job from the HTCondor queue. If a job is stuck or needs to be stopped, this command will terminate it.

#### Debugging Job Submission Issues
```bash
condor_q -better-analyze JobId
```
- `condor_q -better-analyze`: Provides detailed diagnostic information for a job based on its `JobId`, useful for debugging job submission issues (e.g., if a job is not running due to resource constraints).

### 4. File Compression and Decompression

Use `tar` commands to compress and decompress directories or files.

#### Compress a Directory
```bash
tar -czvf filename.tar.gz ALTS/proj_dir/
```
- `tar`: The `tar` utility, used to create or extract files from an archive.
- `-c`: Creates a new archive.
- `-z`: Compresses the archive using gzip.
- `-v`: Shows verbose output, listing the files being processed.
- `-f filename.tar.gz`: Specifies the filename for the compressed archive.
- `ALTS/proj_dir/`: The directory being compressed.

#### Decompress an Archive
```bash
tar -xzvf filename.tar.gz
```
- `-x`: Extracts files from an archive.
- `-z`: Indicates gzip compression.
- `-v`: Shows verbose output, listing files as they are extracted.
- `-f filename.tar.gz`: The archive to be extracted.

### 5. Group GitHub Page

For more resources and collaborative projects, visit the group’s GitHub page:  
[Group GitHub Page](https://github.com/mimrtl)
