Tests with 300 samples, 50000 methylation loci, and 10000 gene expression loci:

<details>
<summary>tecpg run mlr: too much memory</summary>

```python
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0062 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4704 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.2028 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.6795 seconds.
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 600.80128 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
Traceback (most recent call last):
  File "...\Python310\Scripts\tecpg-script.py", line 33, in <module>
    sys.exit(load_entry_point('tecpg', 'console_scripts', 'tecpg')())
  File "...\Python310\Scripts\tecpg-script.py", line 25, in importlib_load_entry_point
    return next(matches).load()
  File "...\Python310\lib\importlib\metadata\__init__.py", line 162, in load
    module = import_module(match.group('module'))
  File "...\Python310\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "...\tecpg\__main__.py", line 9, in <module>
    main()
  File "...\tecpg\__main__.py", line 6, in main
    start()
  File "...\tecpg\cli.py", line 741, in start
    cli(obj={})
  File "...\Python310\lib\site-packages\click\core.py", line 1128, in __call__
    return self.main(*args, **kwargs)
  File "...\Python310\lib\site-packages\click\core.py", line 1053, in main
    rv = self.invoke(ctx)
  File "...\Python310\lib\site-packages\click\core.py", line 1659, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
  File "...\Python310\lib\site-packages\click\core.py", line 1659, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
  File "...\Python310\lib\site-packages\click\core.py", line 1395, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "...\Python310\lib\site-packages\click\core.py", line 754, in invoke
    return __callback(*args, **kwargs)
  File "...\Python310\lib\site-packages\click\decorators.py", line 26, in new_func
    return f(get_current_context(), *args, **kwargs)
  File "...\tecpg\cli.py", line 296, in mlr
    output = regression_full(*args, **logger)
  File "...\tecpg\regression_full.py", line 242, in regression_full
    E = (Y.unsqueeze(1) - X.bmm(B.unsqueeze(2))).squeeze(2)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 58.00 MiB (GPU 0; 8.00 GiB total capacity; 5.67 GiB already allocated; 0 bytes free; 7.24 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

```

</details>

<details>
<summary>tecpg run mlr --cis: 14.2835 seconds</summary>

```python
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0023 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4195 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.0007 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.4226 seconds.
[INFO] No region window provided. Resorting to default.
[INFO] Using default window for cis of 1000000 bases
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 601.28256 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1TIMER] Looped over methylation loci in 8.3837 seconds
[CHUNK1TIMER] Calculated regression_full in 8.3837 seconds
[CHUNK1] Generating dataframe from results...
[CHUNK1TIMER] Finished creating indices in 2.4356 seconds
[CHUNK1TIMER] Finished creating preliminary dataframe in 0.0771 seconds
[CHUNK1TIMER] Created output dataframe in 2.5127 total seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 13.8616 SECONDS
[INFOTIMER] Finished calculating the multiple linear regression in 14.2835 total seconds
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] Saving 1 dataframes...
[INFOTIMER] Saving 1/1: out.csv
[INFOTIMER] Saved 1/1 in 1.0015 seconds
[INFOTIMER] Finished saving 1 dataframes in 1.0016 seconds.

```

</details>

<details>
<summary>tecpg run mlr --distal: 19.8634 seconds</summary>

```python
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0027 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4259 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.1525 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.5811 seconds.
[INFO] No region window provided. Resorting to default.
[INFO] Using default window for distal of 50000 bases
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 601.28256 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1TIMER] Looped over methylation loci in 11.2171 seconds
[CHUNK1TIMER] Calculated regression_full in 11.2171 seconds
[CHUNK1] Generating dataframe from results...
[CHUNK1TIMER] Finished creating indices in 3.0316 seconds
[CHUNK1TIMER] Finished creating preliminary dataframe in 2.0274 seconds
[CHUNK1TIMER] Created output dataframe in 5.0591 total seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 19.4347 SECONDS
[INFOTIMER] Finished calculating the multiple linear regression in 19.8634 total seconds
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] Saving 1 dataframes...
[INFOTIMER] Saving 1/1: out.csv
[INFOTIMER] Saved 1/1 in 67.2793 seconds
[INFOTIMER] Finished saving 1 dataframes in 67.2793 seconds.

```

</details>

<details>
<summary>tecpg run mlr --trans: too much memory</summary>

```python
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0033 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4228 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.0218 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.4479 seconds.
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 601.28256 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
Traceback (most recent call last):
  File "...\Python310\Scripts\tecpg-script.py", line 33, in <module>
    sys.exit(load_entry_point('tecpg', 'console_scripts', 'tecpg')())
  File "...\Python310\Scripts\tecpg-script.py", line 25, in importlib_load_entry_point
    return next(matches).load()
  File "...\Python310\lib\importlib\metadata\__init__.py", line 162, in load
    module = import_module(match.group('module'))
  File "...\Python310\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "...\tecpg\__main__.py", line 9, in <module>
    main()
  File "...\tecpg\__main__.py", line 6, in main
    start()
  File "...\tecpg\cli.py", line 741, in start
    cli(obj={})
  File "...\Python310\lib\site-packages\click\core.py", line 1128, in __call__
    return self.main(*args, **kwargs)
  File "...\Python310\lib\site-packages\click\core.py", line 1053, in main
    rv = self.invoke(ctx)
  File "...\Python310\lib\site-packages\click\core.py", line 1659, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
  File "...\Python310\lib\site-packages\click\core.py", line 1659, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
  File "...\Python310\lib\site-packages\click\core.py", line 1395, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "...\Python310\lib\site-packages\click\core.py", line 754, in invoke
    return __callback(*args, **kwargs)
  File "...\Python310\lib\site-packages\click\decorators.py", line 26, in new_func
    return f(get_current_context(), *args, **kwargs)
  File "...\tecpg\cli.py", line 296, in mlr
    output = regression_full(*args, **logger)
  File "...\tecpg\regression_full.py", line 278, in regression_full
    Y.unsqueeze(1) - X[region_indices].bmm(B.unsqueeze(2))
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 56.00 MiB (GPU 0; 8.00 GiB total capacity; 5.80 GiB already allocated; 0 bytes free; 7.25 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

```

</details>

<details>
<summary>tecpg run mlr -p 0.05: 90.9666 seconds</summary>

```python
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0023 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.3974 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 1.995 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.3948 seconds.
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 600.80128 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1TIMER] Looped over methylation loci in 82.1418 seconds
[CHUNK1TIMER] Calculated regression_full in 82.1419 seconds
[CHUNK1] Generating dataframe from results...
[CHUNK1TIMER] Finished creating indices in 3.2311 seconds
[CHUNK1TIMER] Finished creating preliminary dataframe in 2.3151 seconds
[CHUNK1TIMER] Created output dataframe in 5.5462 total seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 90.5668 SECONDS
[INFOTIMER] Finished calculating the multiple linear regression in 90.9666 total seconds
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] Saving 1 dataframes...
[INFOTIMER] Saving 1/1: out.csv
[INFOTIMER] Saved 1/1 in 83.0815 seconds
[INFOTIMER] Finished saving 1 dataframes in 83.0815 seconds.

```

</details>

<details>
<summary>tecpg run mlr -p 0.05 --cis: 18.5308 seconds</summary>

```python
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0025 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.435 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.1033 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.5409 seconds.
[INFO] No region window provided. Resorting to default.
[INFO] Using default window for cis of 1000000 bases
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 601.28256 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1TIMER] Looped over methylation loci in 12.3876 seconds
[CHUNK1TIMER] Calculated regression_full in 12.3876 seconds
[CHUNK1] Generating dataframe from results...
[CHUNK1TIMER] Finished creating indices in 2.4918 seconds
[CHUNK1TIMER] Finished creating preliminary dataframe in 0.0206 seconds
[CHUNK1TIMER] Created output dataframe in 2.5124 total seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 18.0933 SECONDS
[INFOTIMER] Finished calculating the multiple linear regression in 18.5308 total seconds
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] Saving 1 dataframes...
[INFOTIMER] Saving 1/1: out.csv
[INFOTIMER] Saved 1/1 in 0.0683 seconds
[INFOTIMER] Finished saving 1 dataframes in 0.0683 seconds.

```

</details>

<details>
<summary>tecpg run mlr -p 0.05 --distal: 22.023 seconds</summary>

```python
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0024 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4399 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.1261 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.5684 seconds.
[INFO] No region window provided. Resorting to default.
[INFO] Using default window for distal of 50000 bases
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 601.28256 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1TIMER] Looped over methylation loci in 15.008 seconds
[CHUNK1TIMER] Calculated regression_full in 15.0081 seconds
[CHUNK1] Generating dataframe from results...
[CHUNK1TIMER] Finished creating indices in 3.2812 seconds
[CHUNK1TIMER] Finished creating preliminary dataframe in 0.1547 seconds
[CHUNK1TIMER] Created output dataframe in 3.4359 total seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 21.5807 SECONDS
[INFOTIMER] Finished calculating the multiple linear regression in 22.023 total seconds
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] Saving 1 dataframes...
[INFOTIMER] Saving 1/1: out.csv
[INFOTIMER] Saved 1/1 in 3.6461 seconds
[INFOTIMER] Finished saving 1 dataframes in 3.6461 seconds.

```

</details>

<details>
<summary>tecpg run mlr -p 0.05 --trans: 118.755 seconds</summary>

```python
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0024 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4555 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.1244 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.5825 seconds.
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 601.28256 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1TIMER] Looped over methylation loci in 107.0107 seconds
[CHUNK1TIMER] Calculated regression_full in 107.0107 seconds
[CHUNK1] Generating dataframe from results...
[CHUNK1TIMER] Finished creating indices in 5.8453 seconds
[CHUNK1TIMER] Finished creating preliminary dataframe in 2.2349 seconds
[CHUNK1TIMER] Created output dataframe in 8.0802 total seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 118.297 SECONDS
[INFOTIMER] Finished calculating the multiple linear regression in 118.755 total seconds
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] Saving 1 dataframes...
[INFOTIMER] Saving 1/1: out.csv
[INFOTIMER] Saved 1/1 in 81.6997 seconds
[INFOTIMER] Finished saving 1 dataframes in 81.6998 seconds.

```

</details>

<details>
<summary>tecpg run mlr -g 2000: 541.0004 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 1
[CHUNK1COUNT] Done saving part 2
[CHUNK1COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0024 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.44 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.0365 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.4789 seconds.
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 600.80128 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1] CUDA device memory, chunk 1: 3803.441152 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK1COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 23.3752 seconds. Average chunk time: 23.3752 seconds
[CHUNK1COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 25.6362 seconds. Average chunk time: 24.5057 seconds
[CHUNK1COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 27.086 seconds. Average chunk time: 25.3658 seconds
[CHUNK1COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 27.9078 seconds. Average chunk time: 26.0013 seconds
[CHUNK1COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 29.4497 seconds. Average chunk time: 26.691 seconds
[CHUNK1TIMER] Looped over methylation loci in 136.821 seconds
[CHUNK1TIMER] Calculated regression_full in 136.8211 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 139.8764 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 400.6814 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 541.0004 total seconds

```

</details>

<details>
<summary>tecpg run mlr -g 2000 --cis: 19.0544 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 1
[CHUNK1COUNT] Done saving part 2
[CHUNK1COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0027 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4016 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 1.9615 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.3659 seconds.
[INFO] No region window provided. Resorting to default.
[INFO] Using default window for cis of 1000000 bases
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 601.28256 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1] CUDA device memory, chunk 1: 803.565056 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK1COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 1.9707 seconds. Average chunk time: 1.9707 seconds
[CHUNK1COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 2.3669 seconds. Average chunk time: 2.1688 seconds
[CHUNK1COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 2.3582 seconds. Average chunk time: 2.2319 seconds
[CHUNK1COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 2.3398 seconds. Average chunk time: 2.2589 seconds
[CHUNK1COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.3716 seconds. Average chunk time: 2.2814 seconds
[CHUNK1TIMER] Looped over methylation loci in 12.0669 seconds
[CHUNK1TIMER] Calculated regression_full in 12.0669 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 16.7484 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 1.9015 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 19.0544 total seconds

```

</details>

<details>
<summary>tecpg run mlr -g 2000 --distal: 46.0783 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 1
[CHUNK1COUNT] Done saving part 2
[CHUNK1COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 5
[CHUNK1COUNT] Done saving part 4
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0024 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4252 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 1.9968 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.4245 seconds.
[INFO] No region window provided. Resorting to default.
[INFO] Using default window for distal of 50000 bases
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 601.28256 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1] CUDA device memory, chunk 1: 872.431616 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK1COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 3.279 seconds. Average chunk time: 3.279 seconds
[CHUNK1COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 3.4447 seconds. Average chunk time: 3.3618 seconds
[CHUNK1COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 4.2066 seconds. Average chunk time: 3.6434 seconds
[CHUNK1COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 4.491 seconds. Average chunk time: 3.8553 seconds
[CHUNK1COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 4.4667 seconds. Average chunk time: 3.9776 seconds
[CHUNK1TIMER] Looped over methylation loci in 20.691 seconds
[CHUNK1TIMER] Calculated regression_full in 20.691 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 23.7759 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 21.8747 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 46.0783 total seconds

```

</details>

<details>
<summary>tecpg run mlr -g 2000 --trans: 636.7384 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 1
[CHUNK1COUNT] Done saving part 2
[CHUNK1COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0034 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.5345 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.437 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.9751 seconds.
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 601.28256 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1] CUDA device memory, chunk 1: 3665.18528 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK1COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 29.4672 seconds. Average chunk time: 29.4672 seconds
[CHUNK1COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 32.6803 seconds. Average chunk time: 31.0738 seconds
[CHUNK1COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 34.1074 seconds. Average chunk time: 32.085 seconds
[CHUNK1COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 35.393 seconds. Average chunk time: 32.912 seconds
[CHUNK1COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 36.3125 seconds. Average chunk time: 33.5921 seconds
[CHUNK1TIMER] Looped over methylation loci in 170.8421 seconds
[CHUNK1TIMER] Calculated regression_full in 170.8422 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 174.507 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 461.6933 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 636.7384 total seconds

```

</details>

<details>
<summary>tecpg run mlr -g 2000 -p 0.05: 116.8067 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 1
[CHUNK1COUNT] Done saving part 2
[CHUNK1COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0027 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4478 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.2458 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.6964 seconds.
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 600.80128 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1] CUDA device memory, chunk 1: 905.748992 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK1COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 17.4223 seconds. Average chunk time: 17.4223 seconds
[CHUNK1COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 17.7958 seconds. Average chunk time: 17.609 seconds
[CHUNK1COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 18.0259 seconds. Average chunk time: 17.748 seconds
[CHUNK1COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 18.1583 seconds. Average chunk time: 17.8506 seconds
[CHUNK1COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 18.1529 seconds. Average chunk time: 17.911 seconds
[CHUNK1TIMER] Looped over methylation loci in 90.4651 seconds
[CHUNK1TIMER] Calculated regression_full in 90.4651 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 95.1692 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 21.1869 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 116.8067 total seconds

```

</details>

<details>
<summary>tecpg run mlr -g 2000 -p 0.05 --cis: 22.0005 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 1
[CHUNK1COUNT] Done saving part 2
[CHUNK1COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0024 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4474 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.2313 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.6812 seconds.
[INFO] No region window provided. Resorting to default.
[INFO] Using default window for cis of 1000000 bases
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 601.28256 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1] CUDA device memory, chunk 1: 804.126208 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK1COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 2.8492 seconds. Average chunk time: 2.8492 seconds
[CHUNK1COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 2.9013 seconds. Average chunk time: 2.8753 seconds
[CHUNK1COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 3.1116 seconds. Average chunk time: 2.9541 seconds
[CHUNK1COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 3.256 seconds. Average chunk time: 3.0295 seconds
[CHUNK1COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 3.1961 seconds. Average chunk time: 3.0629 seconds
[CHUNK1TIMER] Looped over methylation loci in 16.0403 seconds
[CHUNK1TIMER] Calculated regression_full in 16.0403 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 19.4634 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 2.0872 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 22.0005 total seconds

```

</details>

<details>
<summary>tecpg run mlr -g 2000 -p 0.05 --distal: 26.1091 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 2
[CHUNK1COUNT] Done saving part 1
[CHUNK1COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0052 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4729 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.0271 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.5053 seconds.
[INFO] No region window provided. Resorting to default.
[INFO] Using default window for distal of 50000 bases
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 601.28256 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1] CUDA device memory, chunk 1: 811.437568 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK1COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 3.5781 seconds. Average chunk time: 3.5781 seconds
[CHUNK1COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 3.7354 seconds. Average chunk time: 3.6567 seconds
[CHUNK1COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 3.8796 seconds. Average chunk time: 3.731 seconds
[CHUNK1COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 4.0086 seconds. Average chunk time: 3.8004 seconds
[CHUNK1COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 3.973 seconds. Average chunk time: 3.8349 seconds
[CHUNK1TIMER] Looped over methylation loci in 19.9012 seconds
[CHUNK1TIMER] Calculated regression_full in 19.9013 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 22.9994 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 2.6316 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 26.1091 total seconds

```

</details>

<details>
<summary>tecpg run mlr -g 2000 -p 0.05 --trans: 142.7494 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 1
[CHUNK1COUNT] Done saving part 2
[CHUNK1COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0053 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.5025 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.4923 seconds
[INFOTIMER] Finished reading 3 dataframes in 3.0002 seconds.
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 601.28256 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1] CUDA device memory, chunk 1: 1165.215744 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK1COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 23.157 seconds. Average chunk time: 23.157 seconds
[CHUNK1COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 23.6645 seconds. Average chunk time: 23.4107 seconds
[CHUNK1COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 23.8837 seconds. Average chunk time: 23.5684 seconds
[CHUNK1COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 23.7507 seconds. Average chunk time: 23.614 seconds
[CHUNK1COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 23.8083 seconds. Average chunk time: 23.6529 seconds
[CHUNK1TIMER] Looped over methylation loci in 119.0898 seconds
[CHUNK1TIMER] Calculated regression_full in 119.0898 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 122.7718 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 19.4697 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 142.7494 total seconds

```

</details>

<details>
<summary>tecpg run mlr -m 10000: 613.4827 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 2
[CHUNK3COUNT] Done saving part 3
[CHUNK4COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0024 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4767 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.2766 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.7557 seconds.
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 120.630272 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1TIMER] Looped over methylation loci in 17.0265 seconds
[CHUNK1TIMER] Calculated regression_full in 17.0265 seconds
[CHUNK1] Generating dataframe from results...
[CHUNK1TIMER] Finished creating indices in 0.677 seconds
[CHUNK1TIMER] Finished creating preliminary dataframe in 7.5022 seconds
[CHUNK1TIMER] Created output dataframe in 8.1792 total seconds
[CHUNK1COUNT] Saving methylation chunk 1/5:
[INFOTIMER] Completed methylation chunk 1/5 in 24.5475 seconds. Average chunk time: 24.5475 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 29.8997 SECONDS
[INFO] STARTING METHYLATION CHUNK 2
[CHUNK2] Running regression_full...
[CHUNK2] CUDA device memory: 120.797184 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK2TIMER] Looped over methylation loci in 16.9543 seconds
[CHUNK2TIMER] Calculated regression_full in 16.9544 seconds
[CHUNK2] Generating dataframe from results...
[CHUNK2TIMER] Finished creating indices in 1.075 seconds
[CHUNK2TIMER] Finished creating preliminary dataframe in 8.2605 seconds
[CHUNK2TIMER] Created output dataframe in 9.3355 total seconds
[CHUNK2COUNT] Saving methylation chunk 2/5:
[INFOTIMER] Completed methylation chunk 2/5 in 26.1028 seconds. Average chunk time: 26.1028 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 2 IN 27.7067 SECONDS
[INFO] STARTING METHYLATION CHUNK 3
[CHUNK3] Running regression_full...
[CHUNK3] CUDA device memory: 120.562688 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK3TIMER] Looped over methylation loci in 16.902 seconds
[CHUNK3TIMER] Calculated regression_full in 16.902 seconds
[CHUNK3] Generating dataframe from results...
[CHUNK3TIMER] Finished creating indices in 1.1095 seconds
[CHUNK3TIMER] Finished creating preliminary dataframe in 9.0961 seconds
[CHUNK3TIMER] Created output dataframe in 10.2056 total seconds
[CHUNK3COUNT] Saving methylation chunk 3/5:
[INFOTIMER] Completed methylation chunk 3/5 in 26.902 seconds. Average chunk time: 26.902 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 3 IN 28.5996 SECONDS
[INFO] STARTING METHYLATION CHUNK 4
[CHUNK4] Running regression_full...
[CHUNK4] CUDA device memory: 120.562688 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK4TIMER] Looped over methylation loci in 17.1249 seconds
[CHUNK4TIMER] Calculated regression_full in 17.125 seconds
[CHUNK4] Generating dataframe from results...
[CHUNK4TIMER] Finished creating indices in 1.1444 seconds
[CHUNK4TIMER] Finished creating preliminary dataframe in 9.8875 seconds
[CHUNK4TIMER] Created output dataframe in 11.0319 total seconds
[CHUNK4COUNT] Saving methylation chunk 4/5:
[INFOTIMER] Completed methylation chunk 4/5 in 27.954 seconds. Average chunk time: 27.954 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 4 IN 29.7701 SECONDS
[INFO] STARTING METHYLATION CHUNK 5
[CHUNK5] Running regression_full...
[CHUNK5] CUDA device memory: 120.562688 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK5TIMER] Looped over methylation loci in 17.1942 seconds
[CHUNK5TIMER] Calculated regression_full in 17.1942 seconds
[CHUNK5] Generating dataframe from results...
[CHUNK5TIMER] Finished creating indices in 1.1721 seconds
[CHUNK5TIMER] Finished creating preliminary dataframe in 10.7162 seconds
[CHUNK5TIMER] Created output dataframe in 11.8883 total seconds
[CHUNK5COUNT] Saving methylation chunk 5/5:
[INFOTIMER] Completed methylation chunk 5/5 in 28.8674 seconds. Average chunk time: 28.8674 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 5 IN 30.803 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 466.2243 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 613.4827 total seconds

```

</details>

<details>
<summary>tecpg run mlr -m 10000 --cis: 58.5784 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 2
[CHUNK3COUNT] Done saving part 3
[CHUNK4COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0076 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4498 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.1919 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.6494 seconds.
[INFO] No region window provided. Resorting to default.
[INFO] Using default window for cis of 1000000 bases
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 120.792064 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1TIMER] Looped over methylation loci in 8.9922 seconds
[CHUNK1TIMER] Calculated regression_full in 8.9923 seconds
[CHUNK1] Generating dataframe from results...
[CHUNK1TIMER] Finished creating indices in 0.5194 seconds
[CHUNK1TIMER] Finished creating preliminary dataframe in 0.0414 seconds
[CHUNK1TIMER] Created output dataframe in 0.5609 total seconds
[CHUNK1COUNT] Saving methylation chunk 1/5:
[INFOTIMER] Completed methylation chunk 1/5 in 8.469 seconds. Average chunk time: 8.469 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 15.5653 SECONDS
[INFO] STARTING METHYLATION CHUNK 2
[CHUNK2] Running regression_full...
[CHUNK2] CUDA device memory: 120.569856 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK2TIMER] Looped over methylation loci in 8.2006 seconds
[CHUNK2TIMER] Calculated regression_full in 8.2007 seconds
[CHUNK2] Generating dataframe from results...
[CHUNK2TIMER] Finished creating indices in 0.5113 seconds
[CHUNK2TIMER] Finished creating preliminary dataframe in 0.0323 seconds
[CHUNK2TIMER] Created output dataframe in 0.5436 total seconds
[CHUNK2COUNT] Saving methylation chunk 2/5:
[INFOTIMER] Completed methylation chunk 2/5 in 8.7063 seconds. Average chunk time: 8.7063 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 2 IN 8.7574 SECONDS
[INFO] STARTING METHYLATION CHUNK 3
[CHUNK3] Running regression_full...
[CHUNK3] CUDA device memory: 120.33536 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK3TIMER] Looped over methylation loci in 8.2621 seconds
[CHUNK3TIMER] Calculated regression_full in 8.2621 seconds
[CHUNK3] Generating dataframe from results...
[CHUNK3TIMER] Finished creating indices in 0.4856 seconds
[CHUNK3TIMER] Finished creating preliminary dataframe in 0.0244 seconds
[CHUNK3TIMER] Created output dataframe in 0.5101 total seconds
[CHUNK3COUNT] Saving methylation chunk 3/5:
[INFOTIMER] Completed methylation chunk 3/5 in 8.7332 seconds. Average chunk time: 8.7332 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 3 IN 8.7839 SECONDS
[INFO] STARTING METHYLATION CHUNK 4
[CHUNK4] Running regression_full...
[CHUNK4] CUDA device memory: 120.33536 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK4TIMER] Looped over methylation loci in 8.1301 seconds
[CHUNK4TIMER] Calculated regression_full in 8.1301 seconds
[CHUNK4] Generating dataframe from results...
[CHUNK4TIMER] Finished creating indices in 0.4834 seconds
[CHUNK4TIMER] Finished creating preliminary dataframe in 0.0259 seconds
[CHUNK4TIMER] Created output dataframe in 0.5094 total seconds
[CHUNK4COUNT] Saving methylation chunk 4/5:
[INFOTIMER] Completed methylation chunk 4/5 in 8.5915 seconds. Average chunk time: 8.5915 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 4 IN 8.652 SECONDS
[INFO] STARTING METHYLATION CHUNK 5
[CHUNK5] Running regression_full...
[CHUNK5] CUDA device memory: 120.33536 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK5TIMER] Looped over methylation loci in 8.2337 seconds
[CHUNK5TIMER] Calculated regression_full in 8.2337 seconds
[CHUNK5] Generating dataframe from results...
[CHUNK5TIMER] Finished creating indices in 0.4932 seconds
[CHUNK5TIMER] Finished creating preliminary dataframe in 0.0249 seconds
[CHUNK5TIMER] Created output dataframe in 0.5181 total seconds
[CHUNK5COUNT] Saving methylation chunk 5/5:
[INFOTIMER] Completed methylation chunk 5/5 in 8.691 seconds. Average chunk time: 8.691 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 5 IN 8.7645 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 7.598 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 58.5784 total seconds

```

</details>

<details>
<summary>tecpg run mlr -m 10000 --distal: 71.1676 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 2
[CHUNK3COUNT] Done saving part 3
[CHUNK4COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0023 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4044 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.0086 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.4154 seconds.
[INFO] No region window provided. Resorting to default.
[INFO] Using default window for distal of 50000 bases
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 120.792064 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1TIMER] Looped over methylation loci in 8.0616 seconds
[CHUNK1TIMER] Calculated regression_full in 8.0616 seconds
[CHUNK1] Generating dataframe from results...
[CHUNK1TIMER] Finished creating indices in 0.588 seconds
[CHUNK1TIMER] Finished creating preliminary dataframe in 0.3508 seconds
[CHUNK1TIMER] Created output dataframe in 0.9389 total seconds
[CHUNK1COUNT] Saving methylation chunk 1/5:
[INFOTIMER] Completed methylation chunk 1/5 in 8.4076 seconds. Average chunk time: 8.4076 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 12.04 SECONDS
[INFO] STARTING METHYLATION CHUNK 2
[CHUNK2] Running regression_full...
[CHUNK2] CUDA device memory: 120.58624 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK2TIMER] Looped over methylation loci in 8.4955 seconds
[CHUNK2TIMER] Calculated regression_full in 8.4955 seconds
[CHUNK2] Generating dataframe from results...
[CHUNK2TIMER] Finished creating indices in 0.6155 seconds
[CHUNK2TIMER] Finished creating preliminary dataframe in 0.4111 seconds
[CHUNK2TIMER] Created output dataframe in 1.0266 total seconds
[CHUNK2COUNT] Saving methylation chunk 2/5:
[INFOTIMER] Completed methylation chunk 2/5 in 9.4704 seconds. Average chunk time: 9.4704 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 2 IN 9.5923 SECONDS
[INFO] STARTING METHYLATION CHUNK 3
[CHUNK3] Running regression_full...
[CHUNK3] CUDA device memory: 120.351744 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK3TIMER] Looped over methylation loci in 8.8924 seconds
[CHUNK3TIMER] Calculated regression_full in 8.8925 seconds
[CHUNK3] Generating dataframe from results...
[CHUNK3TIMER] Finished creating indices in 0.6618 seconds
[CHUNK3TIMER] Finished creating preliminary dataframe in 0.4202 seconds
[CHUNK3TIMER] Created output dataframe in 1.082 total seconds
[CHUNK3COUNT] Saving methylation chunk 3/5:
[INFOTIMER] Completed methylation chunk 3/5 in 9.9302 seconds. Average chunk time: 9.9302 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 3 IN 10.046 SECONDS
[INFO] STARTING METHYLATION CHUNK 4
[CHUNK4] Running regression_full...
[CHUNK4] CUDA device memory: 120.351744 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK4TIMER] Looped over methylation loci in 9.6486 seconds
[CHUNK4TIMER] Calculated regression_full in 9.6486 seconds
[CHUNK4] Generating dataframe from results...
[CHUNK4TIMER] Finished creating indices in 0.7099 seconds
[CHUNK4TIMER] Finished creating preliminary dataframe in 0.4946 seconds
[CHUNK4TIMER] Created output dataframe in 1.2045 total seconds
[CHUNK4COUNT] Saving methylation chunk 4/5:
[INFOTIMER] Completed methylation chunk 4/5 in 10.8066 seconds. Average chunk time: 10.8066 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 4 IN 10.9331 SECONDS
[INFO] STARTING METHYLATION CHUNK 5
[CHUNK5] Running regression_full...
[CHUNK5] CUDA device memory: 120.35072 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK5TIMER] Looped over methylation loci in 9.5428 seconds
[CHUNK5TIMER] Calculated regression_full in 9.5429 seconds
[CHUNK5] Generating dataframe from results...
[CHUNK5TIMER] Finished creating indices in 0.6884 seconds
[CHUNK5TIMER] Finished creating preliminary dataframe in 0.479 seconds
[CHUNK5TIMER] Created output dataframe in 1.1675 total seconds
[CHUNK5COUNT] Saving methylation chunk 5/5:
[INFOTIMER] Completed methylation chunk 5/5 in 10.6621 seconds. Average chunk time: 10.6621 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 5 IN 10.7864 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 17.3629 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 71.1676 total seconds

```

</details>

<details>
<summary>tecpg run mlr -m 10000 --trans: 600.2117 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 2
[CHUNK3COUNT] Done saving part 3
[CHUNK4COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.003 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.5313 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.4975 seconds
[INFOTIMER] Finished reading 3 dataframes in 3.0319 seconds.
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 120.792064 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1TIMER] Looped over methylation loci in 25.3312 seconds
[CHUNK1TIMER] Calculated regression_full in 25.3312 seconds
[CHUNK1] Generating dataframe from results...
[CHUNK1TIMER] Finished creating indices in 1.1042 seconds
[CHUNK1TIMER] Finished creating preliminary dataframe in 6.6189 seconds
[CHUNK1TIMER] Created output dataframe in 7.7232 total seconds
[CHUNK1COUNT] Saving methylation chunk 1/5:
[INFOTIMER] Completed methylation chunk 1/5 in 32.4062 seconds. Average chunk time: 32.4062 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 37.9504 SECONDS
[INFO] STARTING METHYLATION CHUNK 2
[CHUNK2] Running regression_full...
[CHUNK2] CUDA device memory: 120.950784 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK2TIMER] Looped over methylation loci in 25.5431 seconds
[CHUNK2TIMER] Calculated regression_full in 25.5432 seconds
[CHUNK2] Generating dataframe from results...
[CHUNK2TIMER] Finished creating indices in 1.5336 seconds
[CHUNK2TIMER] Finished creating preliminary dataframe in 7.6535 seconds
[CHUNK2TIMER] Created output dataframe in 9.1871 total seconds
[CHUNK2COUNT] Saving methylation chunk 2/5:
[INFOTIMER] Completed methylation chunk 2/5 in 34.5341 seconds. Average chunk time: 34.5341 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 2 IN 35.9816 SECONDS
[INFO] STARTING METHYLATION CHUNK 3
[CHUNK3] Running regression_full...
[CHUNK3] CUDA device memory: 120.717312 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK3TIMER] Looped over methylation loci in 25.8964 seconds
[CHUNK3TIMER] Calculated regression_full in 25.8964 seconds
[CHUNK3] Generating dataframe from results...
[CHUNK3TIMER] Finished creating indices in 1.5651 seconds
[CHUNK3TIMER] Finished creating preliminary dataframe in 7.8476 seconds
[CHUNK3TIMER] Created output dataframe in 9.4127 total seconds
[CHUNK3COUNT] Saving methylation chunk 3/5:
[INFOTIMER] Completed methylation chunk 3/5 in 35.1089 seconds. Average chunk time: 35.1089 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 3 IN 36.5925 SECONDS
[INFO] STARTING METHYLATION CHUNK 4
[CHUNK4] Running regression_full...
[CHUNK4] CUDA device memory: 120.716288 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK4TIMER] Looped over methylation loci in 26.3865 seconds
[CHUNK4TIMER] Calculated regression_full in 26.3865 seconds
[CHUNK4] Generating dataframe from results...
[CHUNK4TIMER] Finished creating indices in 1.6076 seconds
[CHUNK4TIMER] Finished creating preliminary dataframe in 8.539 seconds
[CHUNK4TIMER] Created output dataframe in 10.1467 total seconds
[CHUNK4COUNT] Saving methylation chunk 4/5:
[INFOTIMER] Completed methylation chunk 4/5 in 36.3245 seconds. Average chunk time: 36.3245 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 4 IN 37.8943 SECONDS
[INFO] STARTING METHYLATION CHUNK 5
[CHUNK5] Running regression_full...
[CHUNK5] CUDA device memory: 120.717312 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK5TIMER] Looped over methylation loci in 26.9997 seconds
[CHUNK5TIMER] Calculated regression_full in 26.9997 seconds
[CHUNK5] Generating dataframe from results...
[CHUNK5TIMER] Finished creating indices in 1.7215 seconds
[CHUNK5TIMER] Finished creating preliminary dataframe in 9.4755 seconds
[CHUNK5TIMER] Created output dataframe in 11.1972 total seconds
[CHUNK5COUNT] Saving methylation chunk 5/5:
[INFOTIMER] Completed methylation chunk 5/5 in 37.9581 seconds. Average chunk time: 37.9581 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 5 IN 39.683 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 411.5755 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 600.2117 total seconds

```

</details>

<details>
<summary>tecpg run mlr -m 10000 -p 0.05: 130.7229 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 2
[CHUNK3COUNT] Done saving part 3
[CHUNK4COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0024 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4662 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.0817 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.5504 seconds.
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 120.630272 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1TIMER] Looped over methylation loci in 20.2572 seconds
[CHUNK1TIMER] Calculated regression_full in 20.2572 seconds
[CHUNK1] Generating dataframe from results...
[CHUNK1TIMER] Finished creating indices in 0.5985 seconds
[CHUNK1TIMER] Finished creating preliminary dataframe in 0.4066 seconds
[CHUNK1TIMER] Created output dataframe in 1.0051 total seconds
[CHUNK1COUNT] Saving methylation chunk 1/5:
[INFOTIMER] Completed methylation chunk 1/5 in 20.6388 seconds. Average chunk time: 20.6388 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 25.8516 SECONDS
[INFO] STARTING METHYLATION CHUNK 2
[CHUNK2] Running regression_full...
[CHUNK2] CUDA device memory: 120.414208 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK2TIMER] Looped over methylation loci in 20.1377 seconds
[CHUNK2TIMER] Calculated regression_full in 20.1377 seconds
[CHUNK2] Generating dataframe from results...
[CHUNK2TIMER] Finished creating indices in 0.659 seconds
[CHUNK2TIMER] Finished creating preliminary dataframe in 0.4585 seconds
[CHUNK2TIMER] Created output dataframe in 1.1175 total seconds
[CHUNK2COUNT] Saving methylation chunk 2/5:
[INFOTIMER] Completed methylation chunk 2/5 in 21.2298 seconds. Average chunk time: 21.2298 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 2 IN 21.3314 SECONDS
[INFO] STARTING METHYLATION CHUNK 3
[CHUNK3] Running regression_full...
[CHUNK3] CUDA device memory: 120.179712 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK3TIMER] Looped over methylation loci in 20.2681 seconds
[CHUNK3TIMER] Calculated regression_full in 20.2681 seconds
[CHUNK3] Generating dataframe from results...
[CHUNK3TIMER] Finished creating indices in 0.6688 seconds
[CHUNK3TIMER] Finished creating preliminary dataframe in 0.423 seconds
[CHUNK3TIMER] Created output dataframe in 1.0918 total seconds
[CHUNK3COUNT] Saving methylation chunk 3/5:
[INFOTIMER] Completed methylation chunk 3/5 in 21.3324 seconds. Average chunk time: 21.3324 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 3 IN 21.4332 SECONDS
[INFO] STARTING METHYLATION CHUNK 4
[CHUNK4] Running regression_full...
[CHUNK4] CUDA device memory: 120.18176 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK4TIMER] Looped over methylation loci in 20.4234 seconds
[CHUNK4TIMER] Calculated regression_full in 20.4235 seconds
[CHUNK4] Generating dataframe from results...
[CHUNK4TIMER] Finished creating indices in 0.66 seconds
[CHUNK4TIMER] Finished creating preliminary dataframe in 0.4265 seconds
[CHUNK4TIMER] Created output dataframe in 1.0866 total seconds
[CHUNK4COUNT] Saving methylation chunk 4/5:
[INFOTIMER] Completed methylation chunk 4/5 in 21.4838 seconds. Average chunk time: 21.4838 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 4 IN 21.5894 SECONDS
[INFO] STARTING METHYLATION CHUNK 5
[CHUNK5] Running regression_full...
[CHUNK5] CUDA device memory: 120.18176 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK5TIMER] Looped over methylation loci in 20.2993 seconds
[CHUNK5TIMER] Calculated regression_full in 20.2993 seconds
[CHUNK5] Generating dataframe from results...
[CHUNK5TIMER] Finished creating indices in 0.6556 seconds
[CHUNK5TIMER] Finished creating preliminary dataframe in 0.4648 seconds
[CHUNK5TIMER] Created output dataframe in 1.1204 total seconds
[CHUNK5COUNT] Saving methylation chunk 5/5:
[INFOTIMER] Completed methylation chunk 5/5 in 21.3966 seconds. Average chunk time: 21.3966 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 5 IN 21.4954 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 18.553 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 130.7229 total seconds

```

</details>

<details>
<summary>tecpg run mlr -m 10000 -p 0.05 --cis: 72.3853 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 2
[CHUNK3COUNT] Done saving part 3
[CHUNK4COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0027 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4319 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.0954 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.5301 seconds.
[INFO] No region window provided. Resorting to default.
[INFO] Using default window for cis of 1000000 bases
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 120.792064 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1TIMER] Looped over methylation loci in 11.6337 seconds
[CHUNK1TIMER] Calculated regression_full in 11.6338 seconds
[CHUNK1] Generating dataframe from results...
[CHUNK1TIMER] Finished creating indices in 0.5171 seconds
[CHUNK1TIMER] Finished creating preliminary dataframe in 0.0125 seconds
[CHUNK1TIMER] Created output dataframe in 0.5296 total seconds
[CHUNK1COUNT] Saving methylation chunk 1/5:
[INFOTIMER] Completed methylation chunk 1/5 in 11.5176 seconds. Average chunk time: 11.5176 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 15.4232 SECONDS
[INFO] STARTING METHYLATION CHUNK 2
[CHUNK2] Running regression_full...
[CHUNK2] CUDA device memory: 120.56832 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK2TIMER] Looped over methylation loci in 11.7891 seconds
[CHUNK2TIMER] Calculated regression_full in 11.7891 seconds
[CHUNK2] Generating dataframe from results...
[CHUNK2TIMER] Finished creating indices in 0.52 seconds
[CHUNK2TIMER] Finished creating preliminary dataframe in 0.0106 seconds
[CHUNK2TIMER] Created output dataframe in 0.5306 total seconds
[CHUNK2COUNT] Saving methylation chunk 2/5:
[INFOTIMER] Completed methylation chunk 2/5 in 12.2587 seconds. Average chunk time: 12.2587 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 2 IN 12.3263 SECONDS
[INFO] STARTING METHYLATION CHUNK 3
[CHUNK3] Running regression_full...
[CHUNK3] CUDA device memory: 120.333824 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK3TIMER] Looped over methylation loci in 11.8644 seconds
[CHUNK3TIMER] Calculated regression_full in 11.8644 seconds
[CHUNK3] Generating dataframe from results...
[CHUNK3TIMER] Finished creating indices in 0.5259 seconds
[CHUNK3TIMER] Finished creating preliminary dataframe in 0.0118 seconds
[CHUNK3TIMER] Created output dataframe in 0.5377 total seconds
[CHUNK3COUNT] Saving methylation chunk 3/5:
[INFOTIMER] Completed methylation chunk 3/5 in 12.3674 seconds. Average chunk time: 12.3674 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 3 IN 12.4088 SECONDS
[INFO] STARTING METHYLATION CHUNK 4
[CHUNK4] Running regression_full...
[CHUNK4] CUDA device memory: 120.335872 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK4TIMER] Looped over methylation loci in 11.548 seconds
[CHUNK4TIMER] Calculated regression_full in 11.548 seconds
[CHUNK4] Generating dataframe from results...
[CHUNK4TIMER] Finished creating indices in 0.5259 seconds
[CHUNK4TIMER] Finished creating preliminary dataframe in 0.0106 seconds
[CHUNK4TIMER] Created output dataframe in 0.5365 total seconds
[CHUNK4COUNT] Saving methylation chunk 4/5:
[INFOTIMER] Completed methylation chunk 4/5 in 12.0453 seconds. Average chunk time: 12.0453 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 4 IN 12.0912 SECONDS
[INFO] STARTING METHYLATION CHUNK 5
[CHUNK5] Running regression_full...
[CHUNK5] CUDA device memory: 120.333824 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK5TIMER] Looped over methylation loci in 11.6892 seconds
[CHUNK5TIMER] Calculated regression_full in 11.6893 seconds
[CHUNK5] Generating dataframe from results...
[CHUNK5TIMER] Finished creating indices in 0.5251 seconds
[CHUNK5TIMER] Finished creating preliminary dataframe in 0.0121 seconds
[CHUNK5TIMER] Created output dataframe in 0.5372 total seconds
[CHUNK5COUNT] Saving methylation chunk 5/5:
[INFOTIMER] Completed methylation chunk 5/5 in 12.1923 seconds. Average chunk time: 12.1923 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 5 IN 12.2333 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 7.4678 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 72.3853 total seconds

```

</details>

<details>
<summary>tecpg run mlr -m 10000 -p 0.05 --distal: 70.3481 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 2
[CHUNK3COUNT] Done saving part 3
[CHUNK4COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0023 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.3865 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 1.9813 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.3702 seconds.
[INFO] No region window provided. Resorting to default.
[INFO] Using default window for distal of 50000 bases
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 120.792064 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1TIMER] Looped over methylation loci in 11.6141 seconds
[CHUNK1TIMER] Calculated regression_full in 11.6141 seconds
[CHUNK1] Generating dataframe from results...
[CHUNK1TIMER] Finished creating indices in 0.603 seconds
[CHUNK1TIMER] Finished creating preliminary dataframe in 0.0368 seconds
[CHUNK1TIMER] Created output dataframe in 0.6399 total seconds
[CHUNK1COUNT] Saving methylation chunk 1/5:
[INFOTIMER] Completed methylation chunk 1/5 in 11.6612 seconds. Average chunk time: 11.6612 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 15.2225 SECONDS
[INFO] STARTING METHYLATION CHUNK 2
[CHUNK2] Running regression_full...
[CHUNK2] CUDA device memory: 120.570368 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK2TIMER] Looped over methylation loci in 12.0124 seconds
[CHUNK2TIMER] Calculated regression_full in 12.0124 seconds
[CHUNK2] Generating dataframe from results...
[CHUNK2TIMER] Finished creating indices in 0.6226 seconds
[CHUNK2TIMER] Finished creating preliminary dataframe in 0.0367 seconds
[CHUNK2TIMER] Created output dataframe in 0.6593 total seconds
[CHUNK2COUNT] Saving methylation chunk 2/5:
[INFOTIMER] Completed methylation chunk 2/5 in 12.6357 seconds. Average chunk time: 12.6357 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 2 IN 12.688 SECONDS
[INFO] STARTING METHYLATION CHUNK 3
[CHUNK3] Running regression_full...
[CHUNK3] CUDA device memory: 120.335872 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK3TIMER] Looped over methylation loci in 11.8172 seconds
[CHUNK3TIMER] Calculated regression_full in 11.8172 seconds
[CHUNK3] Generating dataframe from results...
[CHUNK3TIMER] Finished creating indices in 0.6063 seconds
[CHUNK3TIMER] Finished creating preliminary dataframe in 0.0379 seconds
[CHUNK3TIMER] Created output dataframe in 0.6442 total seconds
[CHUNK3COUNT] Saving methylation chunk 3/5:
[INFOTIMER] Completed methylation chunk 3/5 in 12.4237 seconds. Average chunk time: 12.4237 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 3 IN 12.4879 SECONDS
[INFO] STARTING METHYLATION CHUNK 4
[CHUNK4] Running regression_full...
[CHUNK4] CUDA device memory: 120.335872 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK4TIMER] Looped over methylation loci in 11.8703 seconds
[CHUNK4TIMER] Calculated regression_full in 11.8703 seconds
[CHUNK4] Generating dataframe from results...
[CHUNK4TIMER] Finished creating indices in 0.6408 seconds
[CHUNK4TIMER] Finished creating preliminary dataframe in 0.0345 seconds
[CHUNK4TIMER] Created output dataframe in 0.6753 total seconds
[CHUNK4COUNT] Saving methylation chunk 4/5:
[INFOTIMER] Completed methylation chunk 4/5 in 12.5092 seconds. Average chunk time: 12.5092 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 4 IN 12.5613 SECONDS
[INFO] STARTING METHYLATION CHUNK 5
[CHUNK5] Running regression_full...
[CHUNK5] CUDA device memory: 120.335872 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK5TIMER] Looped over methylation loci in 12.3056 seconds
[CHUNK5TIMER] Calculated regression_full in 12.3056 seconds
[CHUNK5] Generating dataframe from results...
[CHUNK5TIMER] Finished creating indices in 0.6202 seconds
[CHUNK5TIMER] Finished creating preliminary dataframe in 0.0403 seconds
[CHUNK5TIMER] Created output dataframe in 0.6605 total seconds
[CHUNK5COUNT] Saving methylation chunk 5/5:
[INFOTIMER] Completed methylation chunk 5/5 in 12.9404 seconds. Average chunk time: 12.9404 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 5 IN 12.9824 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 4.0171 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 70.3481 total seconds

```

</details>

<details>
<summary>tecpg run mlr -m 10000 -p 0.05 --trans: 175.5281 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 2
[CHUNK3COUNT] Done saving part 3
[CHUNK4COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0023 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4197 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.0103 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.4324 seconds.
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 120.792064 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1TIMER] Looped over methylation loci in 28.2227 seconds
[CHUNK1TIMER] Calculated regression_full in 28.2227 seconds
[CHUNK1] Generating dataframe from results...
[CHUNK1TIMER] Finished creating indices in 1.0746 seconds
[CHUNK1TIMER] Finished creating preliminary dataframe in 0.3827 seconds
[CHUNK1TIMER] Created output dataframe in 1.4573 total seconds
[CHUNK1COUNT] Saving methylation chunk 1/5:
[INFOTIMER] Completed methylation chunk 1/5 in 29.0923 seconds. Average chunk time: 29.0923 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 32.7919 SECONDS
[INFO] STARTING METHYLATION CHUNK 2
[CHUNK2] Running regression_full...
[CHUNK2] CUDA device memory: 120.585728 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK2TIMER] Looped over methylation loci in 28.4392 seconds
[CHUNK2TIMER] Calculated regression_full in 28.4392 seconds
[CHUNK2] Generating dataframe from results...
[CHUNK2TIMER] Finished creating indices in 1.1659 seconds
[CHUNK2TIMER] Finished creating preliminary dataframe in 0.4499 seconds
[CHUNK2TIMER] Created output dataframe in 1.6159 total seconds
[CHUNK2COUNT] Saving methylation chunk 2/5:
[INFOTIMER] Completed methylation chunk 2/5 in 30.025 seconds. Average chunk time: 30.025 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 2 IN 30.1337 SECONDS
[INFO] STARTING METHYLATION CHUNK 3
[CHUNK3] Running regression_full...
[CHUNK3] CUDA device memory: 120.351232 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK3TIMER] Looped over methylation loci in 28.4465 seconds
[CHUNK3TIMER] Calculated regression_full in 28.4465 seconds
[CHUNK3] Generating dataframe from results...
[CHUNK3TIMER] Finished creating indices in 1.1578 seconds
[CHUNK3TIMER] Finished creating preliminary dataframe in 0.4521 seconds
[CHUNK3TIMER] Created output dataframe in 1.61 total seconds
[CHUNK3COUNT] Saving methylation chunk 3/5:
[INFOTIMER] Completed methylation chunk 3/5 in 30.0258 seconds. Average chunk time: 30.0258 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 3 IN 30.1278 SECONDS
[INFO] STARTING METHYLATION CHUNK 4
[CHUNK4] Running regression_full...
[CHUNK4] CUDA device memory: 120.351232 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK4TIMER] Looped over methylation loci in 28.2098 seconds
[CHUNK4TIMER] Calculated regression_full in 28.2098 seconds
[CHUNK4] Generating dataframe from results...
[CHUNK4TIMER] Finished creating indices in 1.1591 seconds
[CHUNK4TIMER] Finished creating preliminary dataframe in 0.4342 seconds
[CHUNK4TIMER] Created output dataframe in 1.5933 total seconds
[CHUNK4COUNT] Saving methylation chunk 4/5:
[INFOTIMER] Completed methylation chunk 4/5 in 29.7731 seconds. Average chunk time: 29.7731 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 4 IN 29.8858 SECONDS
[INFO] STARTING METHYLATION CHUNK 5
[CHUNK5] Running regression_full...
[CHUNK5] CUDA device memory: 120.35328 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK5TIMER] Looped over methylation loci in 28.7139 seconds
[CHUNK5TIMER] Calculated regression_full in 28.714 seconds
[CHUNK5] Generating dataframe from results...
[CHUNK5TIMER] Finished creating indices in 1.158 seconds
[CHUNK5TIMER] Finished creating preliminary dataframe in 0.4104 seconds
[CHUNK5TIMER] Created output dataframe in 1.5684 total seconds
[CHUNK5COUNT] Saving methylation chunk 5/5:
[INFOTIMER] Completed methylation chunk 5/5 in 30.2464 seconds. Average chunk time: 30.2464 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 5 IN 30.355 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 21.8118 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 175.5281 total seconds

```

</details>

<details>
<summary>tecpg run mlr -m 10000 -g 2000: 481.9727 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 2
[CHUNK4COUNT] Done saving part 5
[CHUNK5COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 3
[CHUNK5COUNT] Done saving part 1
[CHUNK5COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 2
[CHUNK5COUNT] Done saving part 5
[CHUNK1COUNT] Done saving part 5
[CHUNK5COUNT] Done saving part 3
[CHUNK5COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 1
[CHUNK5COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 2
[CHUNK2COUNT] Done saving part 2
[CHUNK5COUNT] Done saving part 5
[CHUNK5COUNT] Done saving part 3
[CHUNK2COUNT] Done saving part 3
[CHUNK5COUNT] Done saving part 1
[CHUNK5COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 2
[CHUNK5COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0023 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4289 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.0802 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.5115 seconds.
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 120.630272 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1] CUDA device memory, chunk 1: 762.407936 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK1COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 4.8669 seconds. Average chunk time: 4.8669 seconds
[CHUNK1COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 5.3102 seconds. Average chunk time: 5.0886 seconds
[CHUNK1COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 5.4333 seconds. Average chunk time: 5.2035 seconds
[CHUNK1COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 5.8537 seconds. Average chunk time: 5.366 seconds
[CHUNK1COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 6.0598 seconds. Average chunk time: 5.5048 seconds
[CHUNK1TIMER] Looped over methylation loci in 28.435 seconds
[CHUNK1TIMER] Calculated regression_full in 28.435 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 31.6222 SECONDS
[INFO] STARTING METHYLATION CHUNK 2
[CHUNK2] Running regression_full...
[CHUNK2] CUDA device memory: 120.797184 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK2] CUDA device memory, chunk 1: 762.407936 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK2COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 5.8409 seconds. Average chunk time: 5.8409 seconds
[CHUNK2COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 6.211 seconds. Average chunk time: 6.026 seconds
[CHUNK2COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 6.491 seconds. Average chunk time: 6.181 seconds
[CHUNK2COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 6.839 seconds. Average chunk time: 6.3455 seconds
[CHUNK2COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 6.5365 seconds. Average chunk time: 6.3837 seconds
[CHUNK2TIMER] Looped over methylation loci in 31.9705 seconds
[CHUNK2TIMER] Calculated regression_full in 31.9705 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 2 IN 31.9708 SECONDS
[INFO] STARTING METHYLATION CHUNK 3
[CHUNK3] Running regression_full...
[CHUNK3] CUDA device memory: 120.562688 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK3] CUDA device memory, chunk 1: 762.407936 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK3COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 6.4443 seconds. Average chunk time: 6.4443 seconds
[CHUNK3COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 6.5388 seconds. Average chunk time: 6.4915 seconds
[CHUNK3COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 6.0567 seconds. Average chunk time: 6.3466 seconds
[CHUNK3COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 6.059 seconds. Average chunk time: 6.2747 seconds
[CHUNK3COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 7.5033 seconds. Average chunk time: 6.5204 seconds
[CHUNK3TIMER] Looped over methylation loci in 35.0036 seconds
[CHUNK3TIMER] Calculated regression_full in 35.0036 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 3 IN 35.0039 SECONDS
[INFO] STARTING METHYLATION CHUNK 4
[CHUNK4] Running regression_full...
[CHUNK4] CUDA device memory: 120.562688 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK4] CUDA device memory, chunk 1: 762.407936 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK4COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 6.2761 seconds. Average chunk time: 6.2761 seconds
[CHUNK4COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 7.6361 seconds. Average chunk time: 6.9561 seconds
[CHUNK4COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 6.3059 seconds. Average chunk time: 6.7394 seconds
[CHUNK4COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 6.1247 seconds. Average chunk time: 6.5857 seconds
[CHUNK4COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 7.0385 seconds. Average chunk time: 6.6763 seconds
[CHUNK4TIMER] Looped over methylation loci in 37.4977 seconds
[CHUNK4TIMER] Calculated regression_full in 37.4977 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 4 IN 37.498 SECONDS
[INFO] STARTING METHYLATION CHUNK 5
[CHUNK5] Running regression_full...
[CHUNK5] CUDA device memory: 120.562688 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK5] CUDA device memory, chunk 1: 762.407936 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK5COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 7.4304 seconds. Average chunk time: 7.4304 seconds
[CHUNK5COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 7.207 seconds. Average chunk time: 7.3187 seconds
[CHUNK5COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 6.2807 seconds. Average chunk time: 6.9727 seconds
[CHUNK5COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 7.9149 seconds. Average chunk time: 7.2083 seconds
[CHUNK5COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 7.2458 seconds. Average chunk time: 7.2158 seconds
[CHUNK5TIMER] Looped over methylation loci in 38.0982 seconds
[CHUNK5TIMER] Calculated regression_full in 38.0982 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 5 IN 38.0985 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 307.3481 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 481.9727 total seconds

```

</details>

<details>
<summary>tecpg run mlr -m 10000 -g 2000 --cis: 56.6015 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 2
[CHUNK2COUNT] Done saving part 5
[CHUNK4COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 3
[CHUNK3COUNT] Done saving part 1
[CHUNK4COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 5
[CHUNK3COUNT] Done saving part 2
[CHUNK4COUNT] Done saving part 5
[CHUNK1COUNT] Done saving part 4
[CHUNK3COUNT] Done saving part 3
[CHUNK5COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 1
[CHUNK3COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 2
[CHUNK1COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 4
[CHUNK3COUNT] Done saving part 5
[CHUNK5COUNT] Done saving part 3
[CHUNK2COUNT] Done saving part 2
[CHUNK4COUNT] Done saving part 1
[CHUNK5COUNT] Done saving part 4
[CHUNK2COUNT] Done saving part 3
[CHUNK4COUNT] Done saving part 2
[CHUNK5COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0024 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4311 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.0655 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.4991 seconds.
[INFO] No region window provided. Resorting to default.
[INFO] Using default window for cis of 1000000 bases
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 120.792064 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1] CUDA device memory, chunk 1: 163.268608 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK1COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 4.4337 seconds. Average chunk time: 4.4337 seconds
[CHUNK1COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 1.8099 seconds. Average chunk time: 3.1218 seconds
[CHUNK1COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 1.9215 seconds. Average chunk time: 2.7217 seconds
[CHUNK1COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 1.8117 seconds. Average chunk time: 2.4942 seconds
[CHUNK1COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 1.7379 seconds. Average chunk time: 2.3429 seconds
[CHUNK1TIMER] Looped over methylation loci in 12.3698 seconds
[CHUNK1TIMER] Calculated regression_full in 12.3698 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 18.59 SECONDS
[INFO] STARTING METHYLATION CHUNK 2
[CHUNK2] Running regression_full...
[CHUNK2] CUDA device memory: 120.804352 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK2] CUDA device memory, chunk 1: 193.622016 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK2COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 1.8115 seconds. Average chunk time: 1.8115 seconds
[CHUNK2COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 1.9867 seconds. Average chunk time: 1.8991 seconds
[CHUNK2COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 1.8328 seconds. Average chunk time: 1.877 seconds
[CHUNK2COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 1.8583 seconds. Average chunk time: 1.8723 seconds
[CHUNK2COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.0272 seconds. Average chunk time: 1.9033 seconds
[CHUNK2TIMER] Looped over methylation loci in 9.5551 seconds
[CHUNK2TIMER] Calculated regression_full in 9.5551 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 2 IN 9.5553 SECONDS
[INFO] STARTING METHYLATION CHUNK 3
[CHUNK3] Running regression_full...
[CHUNK3] CUDA device memory: 120.804352 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK3] CUDA device memory, chunk 1: 193.622016 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK3COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 1.8424 seconds. Average chunk time: 1.8424 seconds
[CHUNK3COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 1.8446 seconds. Average chunk time: 1.8435 seconds
[CHUNK3COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 1.8766 seconds. Average chunk time: 1.8545 seconds
[CHUNK3COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 1.8947 seconds. Average chunk time: 1.8645 seconds
[CHUNK3COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 1.7709 seconds. Average chunk time: 1.8458 seconds
[CHUNK3TIMER] Looped over methylation loci in 9.2888 seconds
[CHUNK3TIMER] Calculated regression_full in 9.2888 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 3 IN 9.289 SECONDS
[INFO] STARTING METHYLATION CHUNK 4
[CHUNK4] Running regression_full...
[CHUNK4] CUDA device memory: 120.804352 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK4] CUDA device memory, chunk 1: 193.622016 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK4COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 1.8807 seconds. Average chunk time: 1.8807 seconds
[CHUNK4COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 1.7384 seconds. Average chunk time: 1.8095 seconds
[CHUNK4COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 1.7939 seconds. Average chunk time: 1.8043 seconds
[CHUNK4COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 1.7649 seconds. Average chunk time: 1.7945 seconds
[CHUNK4COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 1.8393 seconds. Average chunk time: 1.8034 seconds
[CHUNK4TIMER] Looped over methylation loci in 9.0786 seconds
[CHUNK4TIMER] Calculated regression_full in 9.0786 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 4 IN 9.0789 SECONDS
[INFO] STARTING METHYLATION CHUNK 5
[CHUNK5] Running regression_full...
[CHUNK5] CUDA device memory: 120.804352 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK5] CUDA device memory, chunk 1: 193.622016 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK5COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 1.8598 seconds. Average chunk time: 1.8598 seconds
[CHUNK5COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 1.7919 seconds. Average chunk time: 1.8259 seconds
[CHUNK5COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 1.8327 seconds. Average chunk time: 1.8281 seconds
[CHUNK5COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 1.7999 seconds. Average chunk time: 1.8211 seconds
[CHUNK5COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 1.7846 seconds. Average chunk time: 1.8138 seconds
[CHUNK5TIMER] Looped over methylation loci in 9.1301 seconds
[CHUNK5TIMER] Calculated regression_full in 9.1302 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 5 IN 9.1304 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 0.5242 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 56.6015 total seconds

```

</details>

<details>
<summary>tecpg run mlr -m 10000 -g 2000 --distal: 65.2554 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 2
[CHUNK2COUNT] Done saving part 5
[CHUNK4COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 4
[CHUNK3COUNT] Done saving part 1
[CHUNK4COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 3
[CHUNK3COUNT] Done saving part 2
[CHUNK4COUNT] Done saving part 5
[CHUNK1COUNT] Done saving part 5
[CHUNK3COUNT] Done saving part 3
[CHUNK5COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 2
[CHUNK3COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 2
[CHUNK2COUNT] Done saving part 1
[CHUNK3COUNT] Done saving part 5
[CHUNK5COUNT] Done saving part 3
[CHUNK2COUNT] Done saving part 3
[CHUNK4COUNT] Done saving part 1
[CHUNK5COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 4
[CHUNK4COUNT] Done saving part 2
[CHUNK5COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0037 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4483 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.0715 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.5237 seconds.
[INFO] No region window provided. Resorting to default.
[INFO] Using default window for distal of 50000 bases
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 120.792064 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1] CUDA device memory, chunk 1: 176.710144 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK1COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 1.7062 seconds. Average chunk time: 1.7062 seconds
[CHUNK1COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 1.9454 seconds. Average chunk time: 1.8258 seconds
[CHUNK1COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 2.183 seconds. Average chunk time: 1.9449 seconds
[CHUNK1COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 2.234 seconds. Average chunk time: 2.0172 seconds
[CHUNK1COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.4279 seconds. Average chunk time: 2.0993 seconds
[CHUNK1TIMER] Looped over methylation loci in 11.1387 seconds
[CHUNK1TIMER] Calculated regression_full in 11.1388 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 14.2545 SECONDS
[INFO] STARTING METHYLATION CHUNK 2
[CHUNK2] Running regression_full...
[CHUNK2] CUDA device memory: 120.820736 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK2] CUDA device memory, chunk 1: 193.6384 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK2COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 2.4185 seconds. Average chunk time: 2.4185 seconds
[CHUNK2COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 2.3906 seconds. Average chunk time: 2.4045 seconds
[CHUNK2COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 2.1879 seconds. Average chunk time: 2.3323 seconds
[CHUNK2COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 2.529 seconds. Average chunk time: 2.3815 seconds
[CHUNK2COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.7281 seconds. Average chunk time: 2.4508 seconds
[CHUNK2TIMER] Looped over methylation loci in 12.3094 seconds
[CHUNK2TIMER] Calculated regression_full in 12.3094 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 2 IN 12.3097 SECONDS
[INFO] STARTING METHYLATION CHUNK 3
[CHUNK3] Running regression_full...
[CHUNK3] CUDA device memory: 120.820736 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK3] CUDA device memory, chunk 1: 193.6384 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK3COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 2.2429 seconds. Average chunk time: 2.2429 seconds
[CHUNK3COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 2.0854 seconds. Average chunk time: 2.1642 seconds
[CHUNK3COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 1.9066 seconds. Average chunk time: 2.0783 seconds
[CHUNK3COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 1.9491 seconds. Average chunk time: 2.046 seconds
[CHUNK3COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.1624 seconds. Average chunk time: 2.0693 seconds
[CHUNK3TIMER] Looped over methylation loci in 10.3899 seconds
[CHUNK3TIMER] Calculated regression_full in 10.39 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 3 IN 10.3902 SECONDS
[INFO] STARTING METHYLATION CHUNK 4
[CHUNK4] Running regression_full...
[CHUNK4] CUDA device memory: 120.820736 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK4] CUDA device memory, chunk 1: 193.6384 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK4COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 2.4659 seconds. Average chunk time: 2.4659 seconds
[CHUNK4COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 2.7154 seconds. Average chunk time: 2.5907 seconds
[CHUNK4COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 2.5271 seconds. Average chunk time: 2.5695 seconds
[CHUNK4COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 2.2629 seconds. Average chunk time: 2.4928 seconds
[CHUNK4COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.0806 seconds. Average chunk time: 2.4104 seconds
[CHUNK4TIMER] Looped over methylation loci in 12.106 seconds
[CHUNK4TIMER] Calculated regression_full in 12.1061 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 4 IN 12.1063 SECONDS
[INFO] STARTING METHYLATION CHUNK 5
[CHUNK5] Running regression_full...
[CHUNK5] CUDA device memory: 120.819712 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK5] CUDA device memory, chunk 1: 193.6384 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK5COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 2.0258 seconds. Average chunk time: 2.0258 seconds
[CHUNK5COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 2.0773 seconds. Average chunk time: 2.0516 seconds
[CHUNK5COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 2.0942 seconds. Average chunk time: 2.0658 seconds
[CHUNK5COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 2.4198 seconds. Average chunk time: 2.1543 seconds
[CHUNK5COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.4455 seconds. Average chunk time: 2.2125 seconds
[CHUNK5TIMER] Looped over methylation loci in 11.1054 seconds
[CHUNK5TIMER] Calculated regression_full in 11.1055 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 5 IN 11.1058 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 4.6365 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 65.2554 total seconds

```

</details>

<details>
<summary>tecpg run mlr -m 10000 -g 2000 --trans: 468.8186 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 2
[CHUNK3COUNT] Done saving part 5
[CHUNK5COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 3
[CHUNK3COUNT] Done saving part 1
[CHUNK5COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 4
[CHUNK4COUNT] Done saving part 2
[CHUNK5COUNT] Done saving part 5
[CHUNK1COUNT] Done saving part 5
[CHUNK4COUNT] Done saving part 3
[CHUNK5COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 1
[CHUNK4COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 2
[CHUNK2COUNT] Done saving part 2
[CHUNK4COUNT] Done saving part 5
[CHUNK5COUNT] Done saving part 3
[CHUNK2COUNT] Done saving part 3
[CHUNK5COUNT] Done saving part 1
[CHUNK5COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 2
[CHUNK5COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0023 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.3918 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 1.9861 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.3803 seconds.
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 120.792064 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1] CUDA device memory, chunk 1: 733.981184 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK1COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 6.5853 seconds. Average chunk time: 6.5853 seconds
[CHUNK1COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 7.1553 seconds. Average chunk time: 6.8703 seconds
[CHUNK1COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 7.3927 seconds. Average chunk time: 7.0444 seconds
[CHUNK1COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 7.7864 seconds. Average chunk time: 7.2299 seconds
[CHUNK1COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 8.1083 seconds. Average chunk time: 7.4056 seconds
[CHUNK1TIMER] Looped over methylation loci in 37.9973 seconds
[CHUNK1TIMER] Calculated regression_full in 37.9974 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 41.4138 SECONDS
[INFO] STARTING METHYLATION CHUNK 2
[CHUNK2] Running regression_full...
[CHUNK2] CUDA device memory: 120.950784 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK2] CUDA device memory, chunk 1: 733.991936 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK2COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 7.9118 seconds. Average chunk time: 7.9118 seconds
[CHUNK2COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 8.6234 seconds. Average chunk time: 8.2676 seconds
[CHUNK2COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 8.7434 seconds. Average chunk time: 8.4262 seconds
[CHUNK2COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 9.1329 seconds. Average chunk time: 8.6029 seconds
[CHUNK2COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 8.8901 seconds. Average chunk time: 8.6603 seconds
[CHUNK2TIMER] Looped over methylation loci in 43.3558 seconds
[CHUNK2TIMER] Calculated regression_full in 43.3559 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 2 IN 43.3562 SECONDS
[INFO] STARTING METHYLATION CHUNK 3
[CHUNK3] Running regression_full...
[CHUNK3] CUDA device memory: 120.717312 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK3] CUDA device memory, chunk 1: 733.991936 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK3COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 9.7452 seconds. Average chunk time: 9.7452 seconds
[CHUNK3COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 8.7739 seconds. Average chunk time: 9.2596 seconds
[CHUNK3COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 10.5641 seconds. Average chunk time: 9.6944 seconds
[CHUNK3COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 10.3873 seconds. Average chunk time: 9.8676 seconds
[CHUNK3COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 9.7531 seconds. Average chunk time: 9.8447 seconds
[CHUNK3TIMER] Looped over methylation loci in 53.7495 seconds
[CHUNK3TIMER] Calculated regression_full in 53.7495 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 3 IN 53.75 SECONDS
[INFO] STARTING METHYLATION CHUNK 4
[CHUNK4] Running regression_full...
[CHUNK4] CUDA device memory: 120.716288 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK4] CUDA device memory, chunk 1: 733.991936 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK4COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 10.3409 seconds. Average chunk time: 10.3409 seconds
[CHUNK4COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 8.983 seconds. Average chunk time: 9.662 seconds
[CHUNK4COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 8.9361 seconds. Average chunk time: 9.42 seconds
[CHUNK4COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 10.6661 seconds. Average chunk time: 9.7316 seconds
[CHUNK4COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 9.5011 seconds. Average chunk time: 9.6855 seconds
[CHUNK4TIMER] Looped over methylation loci in 53.054 seconds
[CHUNK4TIMER] Calculated regression_full in 53.054 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 4 IN 53.0543 SECONDS
[INFO] STARTING METHYLATION CHUNK 5
[CHUNK5] Running regression_full...
[CHUNK5] CUDA device memory: 120.717312 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK5] CUDA device memory, chunk 1: 733.991936 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK5COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 10.3538 seconds. Average chunk time: 10.3538 seconds
[CHUNK5COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 9.127 seconds. Average chunk time: 9.7404 seconds
[CHUNK5COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 9.7471 seconds. Average chunk time: 9.7426 seconds
[CHUNK5COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 9.0191 seconds. Average chunk time: 9.5618 seconds
[CHUNK5COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 9.1944 seconds. Average chunk time: 9.4883 seconds
[CHUNK5TIMER] Looped over methylation loci in 51.8963 seconds
[CHUNK5TIMER] Calculated regression_full in 51.8964 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 5 IN 51.8966 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 224.9535 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 468.8186 total seconds

```

</details>

<details>
<summary>tecpg run mlr -m 10000 -g 2000 -p 0.05: 120.7656 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 2
[CHUNK2COUNT] Done saving part 5
[CHUNK4COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 3
[CHUNK3COUNT] Done saving part 1
[CHUNK4COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 4
[CHUNK3COUNT] Done saving part 2
[CHUNK4COUNT] Done saving part 5
[CHUNK1COUNT] Done saving part 5
[CHUNK3COUNT] Done saving part 3
[CHUNK5COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 1
[CHUNK3COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 2
[CHUNK2COUNT] Done saving part 2
[CHUNK3COUNT] Done saving part 5
[CHUNK5COUNT] Done saving part 3
[CHUNK2COUNT] Done saving part 3
[CHUNK4COUNT] Done saving part 1
[CHUNK5COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 4
[CHUNK4COUNT] Done saving part 2
[CHUNK5COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0026 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4123 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 1.9774 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.3924 seconds.
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 120.630272 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1] CUDA device memory, chunk 1: 183.281152 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK1COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 4.2019 seconds. Average chunk time: 4.2019 seconds
[CHUNK1COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 4.1641 seconds. Average chunk time: 4.183 seconds
[CHUNK1COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 4.3452 seconds. Average chunk time: 4.2371 seconds
[CHUNK1COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 4.3927 seconds. Average chunk time: 4.276 seconds
[CHUNK1COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 4.3961 seconds. Average chunk time: 4.3 seconds
[CHUNK1TIMER] Looped over methylation loci in 22.1036 seconds
[CHUNK1TIMER] Calculated regression_full in 22.1036 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 27.9526 SECONDS
[INFO] STARTING METHYLATION CHUNK 2
[CHUNK2] Running regression_full...
[CHUNK2] CUDA device memory: 120.648704 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK2] CUDA device memory, chunk 1: 194.049024 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK2COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 4.4969 seconds. Average chunk time: 4.4969 seconds
[CHUNK2COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 4.3525 seconds. Average chunk time: 4.4247 seconds
[CHUNK2COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 4.4994 seconds. Average chunk time: 4.4496 seconds
[CHUNK2COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 4.4412 seconds. Average chunk time: 4.4475 seconds
[CHUNK2COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 4.4089 seconds. Average chunk time: 4.4398 seconds
[CHUNK2TIMER] Looped over methylation loci in 22.2386 seconds
[CHUNK2TIMER] Calculated regression_full in 22.2386 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 2 IN 22.2388 SECONDS
[INFO] STARTING METHYLATION CHUNK 3
[CHUNK3] Running regression_full...
[CHUNK3] CUDA device memory: 120.648704 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK3] CUDA device memory, chunk 1: 194.049024 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK3COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 4.4038 seconds. Average chunk time: 4.4038 seconds
[CHUNK3COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 4.321 seconds. Average chunk time: 4.3624 seconds
[CHUNK3COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 4.3856 seconds. Average chunk time: 4.3701 seconds
[CHUNK3COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 4.3083 seconds. Average chunk time: 4.3547 seconds
[CHUNK3COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 4.4908 seconds. Average chunk time: 4.3819 seconds
[CHUNK3TIMER] Looped over methylation loci in 21.9489 seconds
[CHUNK3TIMER] Calculated regression_full in 21.949 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 3 IN 21.9493 SECONDS
[INFO] STARTING METHYLATION CHUNK 4
[CHUNK4] Running regression_full...
[CHUNK4] CUDA device memory: 120.650752 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK4] CUDA device memory, chunk 1: 194.051072 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK4COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 4.3957 seconds. Average chunk time: 4.3957 seconds
[CHUNK4COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 4.4575 seconds. Average chunk time: 4.4266 seconds
[CHUNK4COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 4.4795 seconds. Average chunk time: 4.4442 seconds
[CHUNK4COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 4.4441 seconds. Average chunk time: 4.4442 seconds
[CHUNK4COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 4.5129 seconds. Average chunk time: 4.4579 seconds
[CHUNK4TIMER] Looped over methylation loci in 22.3294 seconds
[CHUNK4TIMER] Calculated regression_full in 22.3294 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 4 IN 22.3297 SECONDS
[INFO] STARTING METHYLATION CHUNK 5
[CHUNK5] Running regression_full...
[CHUNK5] CUDA device memory: 120.650752 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK5] CUDA device memory, chunk 1: 194.051072 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK5COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 4.3949 seconds. Average chunk time: 4.3949 seconds
[CHUNK5COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 4.4396 seconds. Average chunk time: 4.4172 seconds
[CHUNK5COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 4.3774 seconds. Average chunk time: 4.4039 seconds
[CHUNK5COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 4.3718 seconds. Average chunk time: 4.3959 seconds
[CHUNK5COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 4.4214 seconds. Average chunk time: 4.401 seconds
[CHUNK5TIMER] Looped over methylation loci in 22.0433 seconds
[CHUNK5TIMER] Calculated regression_full in 22.0433 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 5 IN 22.0437 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 3.8366 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 120.7656 total seconds

```

</details>

<details>
<summary>tecpg run mlr -m 10000 -g 2000 -p 0.05 --cis: 67.5617 seconds</summary>

```python
[CHUNK2COUNT] Done saving part 3
[CHUNK4COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 3
[CHUNK3COUNT] Done saving part 1
[CHUNK4COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 5
[CHUNK3COUNT] Done saving part 2
[CHUNK4COUNT] Done saving part 5
[CHUNK1COUNT] Done saving part 4
[CHUNK3COUNT] Done saving part 3
[CHUNK5COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 1
[CHUNK3COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 2
[CHUNK1COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 4
[CHUNK3COUNT] Done saving part 5
[CHUNK5COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 2
[CHUNK2COUNT] Done saving part 5
[CHUNK4COUNT] Done saving part 1
[CHUNK5COUNT] Done saving part 4
[CHUNK2COUNT] Done saving part 2
[CHUNK4COUNT] Done saving part 2
[CHUNK5COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0025 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4602 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 2.0578 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.5206 seconds.
[INFO] No region window provided. Resorting to default.
[INFO] Using default window for cis of 1000000 bases
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 120.792064 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1] CUDA device memory, chunk 1: 163.549184 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK1COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 2.2187 seconds. Average chunk time: 2.2187 seconds
[CHUNK1COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 2.2659 seconds. Average chunk time: 2.2423 seconds
[CHUNK1COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 2.4621 seconds. Average chunk time: 2.3156 seconds
[CHUNK1COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 2.4466 seconds. Average chunk time: 2.3484 seconds
[CHUNK1COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.8149 seconds. Average chunk time: 2.4417 seconds
[CHUNK1TIMER] Looped over methylation loci in 12.8179 seconds
[CHUNK1TIMER] Calculated regression_full in 12.8179 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 16.0056 SECONDS
[INFO] STARTING METHYLATION CHUNK 2
[CHUNK2] Running regression_full...
[CHUNK2] CUDA device memory: 120.802816 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK2] CUDA device memory, chunk 1: 193.62048 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK2COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 2.7857 seconds. Average chunk time: 2.7857 seconds
[CHUNK2COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 2.4803 seconds. Average chunk time: 2.633 seconds
[CHUNK2COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 2.5566 seconds. Average chunk time: 2.6075 seconds
[CHUNK2COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 2.4216 seconds. Average chunk time: 2.561 seconds
[CHUNK2COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.4667 seconds. Average chunk time: 2.5422 seconds
[CHUNK2TIMER] Looped over methylation loci in 12.7729 seconds
[CHUNK2TIMER] Calculated regression_full in 12.7729 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 2 IN 12.7731 SECONDS
[INFO] STARTING METHYLATION CHUNK 3
[CHUNK3] Running regression_full...
[CHUNK3] CUDA device memory: 120.802816 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK3] CUDA device memory, chunk 1: 193.62048 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK3COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 2.4567 seconds. Average chunk time: 2.4567 seconds
[CHUNK3COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 2.5015 seconds. Average chunk time: 2.4791 seconds
[CHUNK3COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 2.4831 seconds. Average chunk time: 2.4805 seconds
[CHUNK3COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 2.5401 seconds. Average chunk time: 2.4954 seconds
[CHUNK3COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.4824 seconds. Average chunk time: 2.4928 seconds
[CHUNK3TIMER] Looped over methylation loci in 12.5259 seconds
[CHUNK3TIMER] Calculated regression_full in 12.526 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 3 IN 12.5264 SECONDS
[INFO] STARTING METHYLATION CHUNK 4
[CHUNK4] Running regression_full...
[CHUNK4] CUDA device memory: 120.804864 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK4] CUDA device memory, chunk 1: 193.622528 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK4COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 2.459 seconds. Average chunk time: 2.459 seconds
[CHUNK4COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 2.4949 seconds. Average chunk time: 2.477 seconds
[CHUNK4COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 2.4412 seconds. Average chunk time: 2.4651 seconds
[CHUNK4COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 2.5224 seconds. Average chunk time: 2.4794 seconds
[CHUNK4COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.4844 seconds. Average chunk time: 2.4804 seconds
[CHUNK4TIMER] Looped over methylation loci in 12.4635 seconds
[CHUNK4TIMER] Calculated regression_full in 12.4635 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 4 IN 12.4637 SECONDS
[INFO] STARTING METHYLATION CHUNK 5
[CHUNK5] Running regression_full...
[CHUNK5] CUDA device memory: 120.802816 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK5] CUDA device memory, chunk 1: 193.622528 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK5COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 2.6044 seconds. Average chunk time: 2.6044 seconds
[CHUNK5COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 2.5125 seconds. Average chunk time: 2.5584 seconds
[CHUNK5COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 2.5402 seconds. Average chunk time: 2.5524 seconds
[CHUNK5COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 2.5569 seconds. Average chunk time: 2.5535 seconds
[CHUNK5COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.5859 seconds. Average chunk time: 2.56 seconds
[CHUNK5TIMER] Looped over methylation loci in 12.861 seconds
[CHUNK5TIMER] Calculated regression_full in 12.861 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 5 IN 12.8612 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 0.4689 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 67.5617 total seconds

```

</details>

<details>
<summary>tecpg run mlr -m 10000 -g 2000 -p 0.05 --distal: 70.0618 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 3
[CHUNK2COUNT] Done saving part 5
[CHUNK4COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 2
[CHUNK3COUNT] Done saving part 1
[CHUNK4COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 4
[CHUNK3COUNT] Done saving part 2
[CHUNK4COUNT] Done saving part 5
[CHUNK1COUNT] Done saving part 5
[CHUNK3COUNT] Done saving part 3
[CHUNK5COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 1
[CHUNK3COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 2
[CHUNK2COUNT] Done saving part 2
[CHUNK3COUNT] Done saving part 5
[CHUNK5COUNT] Done saving part 3
[CHUNK2COUNT] Done saving part 3
[CHUNK4COUNT] Done saving part 1
[CHUNK5COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 4
[CHUNK4COUNT] Done saving part 2
[CHUNK5COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0023 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4128 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 1.9253 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.3405 seconds.
[INFO] No region window provided. Resorting to default.
[INFO] Using default window for distal of 50000 bases
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 120.792064 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1] CUDA device memory, chunk 1: 164.312064 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK1COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 2.3633 seconds. Average chunk time: 2.3633 seconds
[CHUNK1COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 2.3307 seconds. Average chunk time: 2.347 seconds
[CHUNK1COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 2.5099 seconds. Average chunk time: 2.4013 seconds
[CHUNK1COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 2.5842 seconds. Average chunk time: 2.447 seconds
[CHUNK1COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.7768 seconds. Average chunk time: 2.513 seconds
[CHUNK1TIMER] Looped over methylation loci in 13.1456 seconds
[CHUNK1TIMER] Calculated regression_full in 13.1456 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 16.0522 SECONDS
[INFO] STARTING METHYLATION CHUNK 2
[CHUNK2] Running regression_full...
[CHUNK2] CUDA device memory: 120.804864 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK2] CUDA device memory, chunk 1: 193.622528 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK2COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 2.808 seconds. Average chunk time: 2.808 seconds
[CHUNK2COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 2.6681 seconds. Average chunk time: 2.738 seconds
[CHUNK2COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 2.7412 seconds. Average chunk time: 2.7391 seconds
[CHUNK2COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 3.0832 seconds. Average chunk time: 2.8251 seconds
[CHUNK2COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.6716 seconds. Average chunk time: 2.7944 seconds
[CHUNK2TIMER] Looped over methylation loci in 14.0002 seconds
[CHUNK2TIMER] Calculated regression_full in 14.0003 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 2 IN 14.0005 SECONDS
[INFO] STARTING METHYLATION CHUNK 3
[CHUNK3] Running regression_full...
[CHUNK3] CUDA device memory: 120.804864 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK3] CUDA device memory, chunk 1: 193.622528 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK3COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 2.8179 seconds. Average chunk time: 2.8179 seconds
[CHUNK3COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 2.5862 seconds. Average chunk time: 2.7021 seconds
[CHUNK3COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 2.5974 seconds. Average chunk time: 2.6672 seconds
[CHUNK3COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 2.4695 seconds. Average chunk time: 2.6178 seconds
[CHUNK3COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.4542 seconds. Average chunk time: 2.5851 seconds
[CHUNK3TIMER] Looped over methylation loci in 12.9516 seconds
[CHUNK3TIMER] Calculated regression_full in 12.9517 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 3 IN 12.9519 SECONDS
[INFO] STARTING METHYLATION CHUNK 4
[CHUNK4] Running regression_full...
[CHUNK4] CUDA device memory: 120.804864 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK4] CUDA device memory, chunk 1: 193.622528 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK4COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 2.5416 seconds. Average chunk time: 2.5416 seconds
[CHUNK4COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 2.7962 seconds. Average chunk time: 2.6689 seconds
[CHUNK4COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 2.5747 seconds. Average chunk time: 2.6375 seconds
[CHUNK4COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 2.572 seconds. Average chunk time: 2.6211 seconds
[CHUNK4COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.5175 seconds. Average chunk time: 2.6004 seconds
[CHUNK4TIMER] Looped over methylation loci in 13.0271 seconds
[CHUNK4TIMER] Calculated regression_full in 13.0271 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 4 IN 13.0273 SECONDS
[INFO] STARTING METHYLATION CHUNK 5
[CHUNK5] Running regression_full...
[CHUNK5] CUDA device memory: 120.804864 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK5] CUDA device memory, chunk 1: 193.622528 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK5COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 2.5419 seconds. Average chunk time: 2.5419 seconds
[CHUNK5COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 2.4902 seconds. Average chunk time: 2.516 seconds
[CHUNK5COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 2.5885 seconds. Average chunk time: 2.5402 seconds
[CHUNK5COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 2.5886 seconds. Average chunk time: 2.5523 seconds
[CHUNK5COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 2.7456 seconds. Average chunk time: 2.5909 seconds
[CHUNK5TIMER] Looped over methylation loci in 12.9813 seconds
[CHUNK5TIMER] Calculated regression_full in 12.9813 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 5 IN 12.9815 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 0.6331 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 70.0618 total seconds

```

</details>

<details>
<summary>tecpg run mlr -m 10000 -g 2000 -p 0.05 --trans: 162.4002 seconds</summary>

```python
[CHUNK1COUNT] Done saving part 2
[CHUNK2COUNT] Done saving part 5
[CHUNK4COUNT] Done saving part 3
[CHUNK1COUNT] Done saving part 3
[CHUNK3COUNT] Done saving part 1
[CHUNK4COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 4
[CHUNK3COUNT] Done saving part 2
[CHUNK4COUNT] Done saving part 5
[CHUNK1COUNT] Done saving part 5
[CHUNK3COUNT] Done saving part 3
[CHUNK5COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 1
[CHUNK3COUNT] Done saving part 4
[CHUNK5COUNT] Done saving part 2
[CHUNK2COUNT] Done saving part 2
[CHUNK3COUNT] Done saving part 5
[CHUNK5COUNT] Done saving part 3
[CHUNK2COUNT] Done saving part 3
[CHUNK4COUNT] Done saving part 1
[CHUNK5COUNT] Done saving part 4
[CHUNK1COUNT] Done saving part 1
[CHUNK2COUNT] Done saving part 4
[CHUNK4COUNT] Done saving part 2
[CHUNK5COUNT] Done saving part 5
[INFO] CUDA GPU detected. This device supports CUDA.
[INFO] Reading 3 dataframes...
[INFOTIMER] Reading 1/3: C.csv
[INFO] Reading csv file ...\working\data\C.csv with separator ,
[INFOTIMER] Read 1/3 in 0.0023 seconds
[INFOTIMER] Reading 2/3: G.csv
[INFO] Reading csv file ...\working\data\G.csv with separator ,
[INFOTIMER] Read 2/3 in 0.4104 seconds
[INFOTIMER] Reading 3/3: M.csv
[INFO] Reading csv file ...\working\data\M.csv with separator ,
[INFOTIMER] Read 3/3 in 1.9389 seconds
[INFOTIMER] Finished reading 3 dataframes in 2.3517 seconds.
[INFO] Initializing regression variables
[INFO] Use CPU not supplied. Checking if CUDA is available.
[INFO] Using CUDA
[INFO] Running with 296 degrees of freedom
[INFO] Initializing output directory
[INFO] Removing directory ...\working\output...
[INFO] Creating directory ...\working\output...
[INFO] STARTING METHYLATION CHUNK 1
[CHUNK1] Running regression_full...
[CHUNK1] CUDA device memory: 120.792064 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK1] CUDA device memory, chunk 1: 234.598912 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK1COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 5.8569 seconds. Average chunk time: 5.8569 seconds
[CHUNK1COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 6.0095 seconds. Average chunk time: 5.9332 seconds
[CHUNK1COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 6.0915 seconds. Average chunk time: 5.9859 seconds
[CHUNK1COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 6.1201 seconds. Average chunk time: 6.0195 seconds
[CHUNK1COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 6.2658 seconds. Average chunk time: 6.0687 seconds
[CHUNK1TIMER] Looped over methylation loci in 30.9268 seconds
[CHUNK1TIMER] Calculated regression_full in 30.9268 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 1 IN 33.8484 SECONDS
[INFO] STARTING METHYLATION CHUNK 2
[CHUNK2] Running regression_full...
[CHUNK2] CUDA device memory: 120.820224 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK2] CUDA device memory, chunk 1: 234.601472 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK2COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 6.3357 seconds. Average chunk time: 6.3357 seconds
[CHUNK2COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 6.2016 seconds. Average chunk time: 6.2686 seconds
[CHUNK2COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 6.2242 seconds. Average chunk time: 6.2538 seconds
[CHUNK2COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 6.2158 seconds. Average chunk time: 6.2443 seconds
[CHUNK2COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 6.1359 seconds. Average chunk time: 6.2226 seconds
[CHUNK2TIMER] Looped over methylation loci in 31.1498 seconds
[CHUNK2TIMER] Calculated regression_full in 31.1498 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 2 IN 31.1501 SECONDS
[INFO] STARTING METHYLATION CHUNK 3
[CHUNK3] Running regression_full...
[CHUNK3] CUDA device memory: 120.820224 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK3] CUDA device memory, chunk 1: 234.60608 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK3COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 6.1893 seconds. Average chunk time: 6.1893 seconds
[CHUNK3COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 6.1795 seconds. Average chunk time: 6.1844 seconds
[CHUNK3COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 6.3938 seconds. Average chunk time: 6.2542 seconds
[CHUNK3COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 6.352 seconds. Average chunk time: 6.2787 seconds
[CHUNK3COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 6.3183 seconds. Average chunk time: 6.2866 seconds
[CHUNK3TIMER] Looped over methylation loci in 31.472 seconds
[CHUNK3TIMER] Calculated regression_full in 31.472 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 3 IN 31.4723 SECONDS
[INFO] STARTING METHYLATION CHUNK 4
[CHUNK4] Running regression_full...
[CHUNK4] CUDA device memory: 120.820224 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK4] CUDA device memory, chunk 1: 234.646528 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK4COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 6.3612 seconds. Average chunk time: 6.3612 seconds
[CHUNK4COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 6.1302 seconds. Average chunk time: 6.2457 seconds
[CHUNK4COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 6.121 seconds. Average chunk time: 6.2041 seconds
[CHUNK4COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 6.1961 seconds. Average chunk time: 6.2021 seconds
[CHUNK4COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 6.1208 seconds. Average chunk time: 6.1858 seconds
[CHUNK4TIMER] Looped over methylation loci in 30.9691 seconds
[CHUNK4TIMER] Calculated regression_full in 30.9691 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 4 IN 30.9694 SECONDS
[INFO] STARTING METHYLATION CHUNK 5
[CHUNK5] Running regression_full...
[CHUNK5] CUDA device memory: 120.822272 MB allocated by constants out of 8589.47584 MB total
[INFO] Calculating regression...
[CHUNK5] CUDA device memory, chunk 1: 234.646528 MB allocated out of 8589.47584 MB total. If needed, increase --loci-per-chunk accordingly
[CHUNK5COUNT] Saving part 1/5:
[INFOTIMER] Completed chunk 1/5 in 6.1496 seconds. Average chunk time: 6.1496 seconds
[CHUNK5COUNT] Saving part 2/5:
[INFOTIMER] Completed chunk 2/5 in 6.1596 seconds. Average chunk time: 6.1546 seconds
[CHUNK5COUNT] Saving part 3/5:
[INFOTIMER] Completed chunk 3/5 in 6.1624 seconds. Average chunk time: 6.1572 seconds
[CHUNK5COUNT] Saving part 4/5:
[INFOTIMER] Completed chunk 4/5 in 6.1901 seconds. Average chunk time: 6.1654 seconds
[CHUNK5COUNT] Saving part 5/5:
[INFOTIMER] Completed chunk 5/5 in 6.1264 seconds. Average chunk time: 6.1576 seconds
[CHUNK5TIMER] Looped over methylation loci in 30.8268 seconds
[CHUNK5TIMER] Calculated regression_full in 30.8269 seconds
[INFOTIMER] FINISHED METHYLATION CHUNK 5 IN 30.8271 SECONDS
[INFOTIMER] Waiting for chunks to save...
[INFOTIMER] Finished waiting for chunks to save in 3.7201 seconds
[INFOTIMER] Finished calculating the multiple linear regression in 162.4002 total seconds

```

</details>
