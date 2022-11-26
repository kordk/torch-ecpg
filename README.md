# tecpg

Python eCpG mapper with CLI using pytorch

## Installation

Pip install from github using `git+https://`.

```bash
pip install git+https://github.com/kordk/torch-ecpg.git
```

If you want to be able to edit the code for debugging and development, install in editable mode and do not remove the directory.

```bash
cd [path/to/code/directory]
git clone https://github.com/kordk/torch-ecpg.git
cd tecpg
pip install --editable .
```

`tecpg` is an entry point in the command line than calls the root CLI function. If the installation was successful, running `tecpg --help` should provide help with the command line interface.

**If you have issues with using `pip` in the command line, try `python -m pip` or `python3 -m pip`**

## Dependencies

Stored in `tecpg/requirements.txt`:

```
click>=8.0.3
colorama>=0.4.4
matplotlib>=3.5.1
numpy>=1.22.4
pandas>=1.3.5
requests>=2.26.0
scipy>=1.7.2
setuptools>=63.3.0
torch>=1.11.0+cu113
```

## Conditions

Pytorch will run on the CPU or CUDA-supported GPU. A GPU is generally much faster. The device is automatically selected based on `torch.cuda.is_available()`.

Python 3.10 or higher is required.

## Notes

The files `pearson_single.py` and `test_pearson_single.py` are for testing purposes only. They also serve to illustrate the major implementations of the pearson correlation coefficient function. The file `pearson_full.py` contains the optimized versions of the pearson correlation coefficient function that operate on the entire input matrix. These should be used in most cases and are the only options available in the CLI.

## Input formats

There are three main inputs for most functions: methylation beta values, gene expression values, and covariates. Some functions, like the pearson correlation coefficient functions, only take in methylation and gene expression data.

Methylation and gene expression matrices are provided as csv or tsv files. The **column names are the sample ids** and the **row names are the gene or methylation site ids**. The values are either the gene expression for the sample at a particular gene id or the methylation beta value for a sample at a particular methylation site id.

Covariates are stored in a csv or tsv file, with the **columns being the covariates** and the **rows being the sample ids**. The values represent the value of the covariate for each sample.

### Covariate preprocessing

Covariate preprocessing is expected of the user. It is expected that categorical values are preprocessed to be split into n - 1 binary columns. **All data must be numerical.** Binary categorical data should be converted to 0 and 1. Ordinal values should be preprocessed as desired but needs to be numerical.

Examples:

<details>
<summary>data/M.csv</summary>

<table>
<tr>
<th>Methylation ID</th><th>5881</th><th>5896</th><th>5915</th><th>5949</th><th>5965</th><th>5988</th>
</tr>
<tr><th>cg00000029</th><td>0.551142626425936</td><td>0.606679809418831</td><td>0.593760482022385</td><td>0.554829598676022</td><td>0.653260367643675</td><td>0.608085832424757</td>
<tr><th>cg00000108</th><td>0.998563692332771</td><td>0.9979593001545</td><td>0.997893371350954</td><td>0.997293677663346</td><td>0.995090033303771</td><td>0.998615804087594</td>
<tr><th>cg00000109</th><td>0.992806501740033</td><td>0.985118377090081</td><td>0.992321161423107</td><td>0.999350415390975</td><td>0.987700359037286</td><td>0.989837525369162</td>
<tr><th>cg00000165</th><td>0.266529984719736</td><td>0.159711109475489</td><td>0.145981687514545</td><td>0.100000350688528</td><td>0.186060083535488</td><td>0.206889623568409</td>
<tr><th>cg00000236</th><td>0.812799925026805</td><td>0.897011511592051</td><td>0.908067942964869</td><td>0.863719773724759</td><td>0.86472623747957</td><td>0.901683566050475</td>
<tr><th>cg00000289</th><td>0.799775664748878</td><td>0.7998679244224</td><td>0.793414346221782</td><td>0.778572418611009</td><td>0.811938518189902</td><td>0.793507626962159</td>
<tr><th>cg00000292</th><td>0.953231435096254</td><td>0.938383330335049</td><td>0.915259919658409</td><td>0.92810976339309</td><td>0.921614041219709</td><td>0.961061233113192</td>
<tr><th>cg00000321</th><td>0.402541802839615</td><td>0.246483561918606</td><td>0.269776112390868</td><td>0.156315665588217</td><td>0.29074031330004</td><td>0.268051042501277</td>
<tr><th>cg00000363</th><td>0.351440801449114</td><td>0.388650885657661</td><td>0.338968413156223</td><td>0.323264024048326</td><td>0.307826692557988</td><td>0.391903676457131</td>
<tr><th>cg00000622</th><td>0.006123901103999</td><td>0.0042042851381022</td><td>0.0078936545402247</td><td>0.0030549368245614</td><td>0.0090032960387964</td><td>0.0123730634317925</td>
<tr><th>cg00000658</th><td>0.97236198533073</td><td>0.940943221689518</td><td>0.960950626130923</td><td>0.972090418533777</td><td>0.964900056587945</td><td>0.955162502333346</td>
<tr><th>cg00000714</th><td>0.153027877334071</td><td>0.202331113857809</td><td>0.199132391280944</td><td>0.129903643858353</td><td>0.148630614399603</td><td>0.150078405435778</td>
<tr><th>cg00000721</th><td>0.991601150232433</td><td>0.998404687879398</td><td>0.988131334474828</td><td>0.998983884129865</td><td>0.998792503281812</td><td>0.996650916895041</td>
<tr><th>cg00000734</th><td>0.0188531428701418</td><td>0.0367560070824414</td><td>0.0217869037614336</td><td>0.0176911178988943</td><td>0.0195149047534552</td><td>0.0218093808733567</td>
<tr><th>cg00000769</th><td>0.0252193945550211</td><td>0.0444667969594232</td><td>0.0331255360470739</td><td>0.0405055370452657</td><td>0.0265462252960827</td><td>0.0214920723339295</td>
</table>

</details>

<details>
<summary>data/G.csv</summary>

<table>
<tr><th>Gene ID</th><th>5881</th><th>5896</th><th>5915</th><th>5949</th><th>5965</th><th>5988</th></tr>
<tr><th>ILMN_1762337</th><td>43.10106</td><td>48.30485</td><td>37.49239</td><td>43.99564</td><td>39.44977</td><td>45.18019</td></tr>
<tr><th>ILMN_2055271</th><td>61.09617</td><td>61.84258</td><td>47.78094</td><td>49.32763</td><td>50.70347</td><td>48.43206</td></tr>
<tr><th>ILMN_1736007</th><td>51.30634</td><td>45.80393</td><td>45.43285</td><td>40.39254</td><td>34.89904</td><td>44.05256</td></tr>
<tr><th>ILMN_2383229</th><td>48.15523</td><td>42.69902</td><td>35.71749</td><td>39.52501</td><td>46.40649</td><td>43.0822</td></tr>
<tr><th>ILMN_1806310</th><td>42.00099</td><td>53.43919</td><td>41.79802</td><td>44.9275</td><td>44.51156</td><td>38.64714</td></tr>
<tr><th>ILMN_1779670</th><td>55.97569</td><td>65.66503</td><td>51.50679</td><td>42.56688</td><td>54.42937</td><td>45.65614</td></tr>
<tr><th>ILMN_1653355</th><td>63.82549</td><td>83.52248</td><td>73.94881</td><td>65.28152</td><td>62.74917</td><td>62.30242</td></tr>
<tr><th>ILMN_1717783</th><td>38.80901</td><td>46.2599</td><td>41.70654</td><td>38.58479</td><td>36.94595</td><td>41.42864</td></tr>
<tr><th>ILMN_1705025</th><td>50.55648</td><td>52.62954</td><td>45.416</td><td>41.77805</td><td>45.90888</td><td>46.31426</td></tr>
<tr><th>ILMN_1814316</th><td>45.12116</td><td>47.13919</td><td>40.49826</td><td>42.99829</td><td>45.22402</td><td>40.86175</td></tr>
<tr><th>ILMN_2359168</th><td>42.97596</td><td>51.4101</td><td>42.49061</td><td>44.52789</td><td>37.77698</td><td>45.15334</td></tr>
<tr><th>ILMN_1731507</th><td>37.32842</td><td>39.2299</td><td>36.14508</td><td>39.29216</td><td>32.67036</td><td>38.08112</td></tr>
<tr><th>ILMN_1787689</th><td>37.14457</td><td>49.44761</td><td>41.09594</td><td>36.4364</td><td>38.90872</td><td>37.4399</td></tr>
<tr><th>ILMN_1745607</th><td>43.92733</td><td>45.20126</td><td>43.44871</td><td>35.70976</td><td>40.1534</td><td>38.89397</td></tr>
<tr><th>ILMN_2136495</th><td>48.76113</td><td>43.45613</td><td>40.16721</td><td>40.98922</td><td>40.94952</td><td>42.08607</td></tr>
<tr><th>ILMN_1668111</th><td>39.2202</td><td>42.98085</td><td>39.08544</td><td>37.5288</td><td>34.42897</td><td>43.60989</td></tr>
</table>

</details>

<details>
<summary>data/C.csv</summary>

<table>
<tr><th>Sample ID</th><th>Sex</th><th>age</th><th>dna methylation-predicted age</th><th>cd8 t cells</th><th>cd4 t cells</th><th>natural killer cells</th><th>b cells</th><th>monocytes</th><th>granulocytes</th></tr>
<tr><th>5881</th><td>0</td><td>44</td><td>49.05389863</td><td>0.03922</td><td>0.21532</td><td>0.02287</td><td>0.07433</td><td>0.08114</td><td>0.58146</td></tr>
<tr><th>5896</th><td>0</td><td>50</td><td>31.98340803</td><td>0.00779</td><td>0.22337</td><td>0.07083</td><td>0.03843</td><td>0.06435</td><td>0.60609</td></tr>
<tr><th>5915</th><td>1</td><td>52</td><td>42.96901918</td><td>0</td><td>0.13076</td><td>0.10282</td><td>0.06882</td><td>0.07651</td><td>0.6344</td></tr>
<tr><th>5949</th><td>0</td><td>56</td><td>38.36846554</td><td>0.05287</td><td>0.17147</td><td>0.08456</td><td>0.04391</td><td>0.06156</td><td>0.61011</td></tr>
<tr><th>5965</th><td>0</td><td>74</td><td>39.49148329</td><td>0.0782</td><td>0.18076</td><td>0.24658</td><td>0.01991</td><td>0.0622</td><td>0.40619</td></tr>
<tr><th>5988</th><td>1</td><td>47</td><td>50.5973987</td><td>0.08049</td><td>0.23979</td><td>0.0807</td><td>0.08356</td><td>0.06942</td><td>0.469</td></tr>
</table>

</details>

<br>

## Running on the CPU

By default, the selected algorithm will run with a CUDA enabled device, if available, and otherwise on the CPU. Information about the chosen device and CUDA availability is logged. If you would like to explicitly run on the CPU, use the --cpu-threads or -t option followed by the number of threads of the CPU to use. The default is 0 threads, meaning a CUDA enabled device is used if available.

For example, to use a command with 4 threads, run:

```
tecpg -t 4 run [COMMAND]
```

## Chunking

To avoid memory limits, some algorithms calculate and save results in chunks.
--chunks: total number of chunks to divide the input
--save-chunks: the number of compute chunks per save
--chunk-size: the number of computations to perform per chunk

## Output filtration

P-value filtering is implemented for the multiple linear regression. Use the --p-thresh option to specify a minimum gene expression p-value to be included in the output. This will force p-value calculation, even if p-values are not included in the output.

# Usage

## Python Module

The python module provides the most flexibility, power, and automation, at the cost of setup speed.

### Tips

- Initialize the logger at the top of the root file (logger = Logger())
- Pass the logger to functions using **logger: function(args, **logger)
- The letters M, G, and C mean methylation beta values, gene expression values, and covariates
- Correlation, or corr, refers to the pearson correlation coefficient
- MLR or regression refer to the multiple linear regression algorithm

### Examples

Implementation of **tecpg run corr**:

```py
import os
from tecpg.test_data import generate_data
from tecpg.pearson_full import pearson_full_tensor
from tecpg.logger import Logger

parent_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(parent_path, 'out.csv')

# Initialize the logger in debug mode
logger = Logger(debug=True)

# Generate data with 100 samples, 300 methylation rows, and 300 gene
# expression rows.
M, G, _ = generate_data(100, 300, 300)

# Pass logger to supported functions with **logger
corr = pearson_full_tensor(M, G, **logger)

# Corr is the correlation dataframe. Save it with corr.to_csv(file_path)
corr.to_csv(file_path)
```

## CLI

The command line interface is the simplest and most convenient way of getting started with tecpg. The CLI is minimal and only includes the main functions.

Setup:

```bash
> cd Desktop
> tecpg init testing -y # Initializes Desktop/testing as the testing environment
> cd testing
> tecpg data dummy
Samples: 100
Meth rows: 300
Gene rows: 300
```

<details>
<summary> Correlation Examples </summary>

Running a simple correlation on dummy data:

```bash
> tecpg run corr # Because no chunk counts were provided, defaults to pearson_full_tensor (better for small inputs)
> ls output
out.csv
```

Running chunked correlations:

```bash
> rm output/out.csv
> tecpg run corr -c 10 # Chunks provided, runs pearson_chunk_tensor (better for large inputs)
> ls output
out.csv
> rm output/out.csv
> tecpg run corr -c 10 -s 2 # Save chunks provided, runs pearson_chunk_save_tensor (better for huge outputs, 20gb or more). Saves a new file every 2 chunks.
> ls output # 10 chunks are saved 2 at a time for a total of 5 save files
1.csv # Chunks 1 and 2
2.csv # Chunks 3 and 4
3.csv # Chunks 5 and 6
4.csv # Chunks 7 and 8
5.csv # Chunks 9 and 10
```

Running correlation on Grady Trauma Project data:

```bash
> rm output/*
> tecpg data gtp -y
> tecpg run corr -c 1000 -s 10 # Could take a few hours and over 250gb of space
> ls output
1.csv # Chunks 1 to 10
2.csv # Chunks 11 to 20
...
99.csv # Chunks 981 990
100.csv # Chunks 991 to 1000
```

</details>

<details>
<summary> Multiple Linear Regression Examples </summary>

Running a multiple linear regression in the current directory:

```bash
> tecpg run mlr
```

</details>

# CLI Structure

```
tecpg : The root cli group
 => data : Base group for running algorithms.
     => dummy : Generates dummy data.
     => gtp : Downloads and extracts GTP data.
 => run : Base group for running algorithms.
     => corr : Calculate the pearson correlation coefficient.
 => init : Creates and initializes directory.
```

### CLI Commands

```
tecpg [--root-path -r DIRECTORY] [--input-dir -i DIRECTORY] [--output-dir -o DIRECTORY] [--meth_file -m FILE] [--gene_file -g FILE] [--covar_file -c FILE] [--output_file -f FILE] [--cpu-threads -t INTEGER] [--verbosity -v] [--debug -d] [--log-dir -l DIRECTORY] [--no-log-file -n] [--help]
```

```
tecpg data [--help]
```

```
tecpg data dummy [--samples -s INTEGER] [--meth-rows -m INTEGER] [--gene-rows -g INTEGER] [--help]
```

```
tecpg data gtp [--gtp-dir -g DIRECTORY] [--yes -y] [--help]
```

```
tecpg run [--help]
```

```
tecpg run corr [--chunks -c INTEGER] [--save-chunks -s INTEGER] [--flatten -f] [--help]
```

```
tecpg run mlr [--regressions-per-chunk -r INTEGER] [--p-thresh -p FLOAT] [--full-output -f] [--no-est] [--no-err] [--no-t] [--no-p] [--help]
```

### Documentation

<details>
<summary><b>tecpg</b></summary>

```

The root cli group

Options:
    -r, --root-path DIRECTORY   [default: .]
    -i, --input-dir DIRECTORY   [default: data]
    -o, --output-dir DIRECTORY  [default: output]
    -m, --meth-file FILE        [default: M.csv]
    -g, --gene-file FILE        [default: G.csv]
    -c, --covar-file FILE       [default: C.csv]
    -f, --output-file FILE      [default: out.csv]
    -t, --cpu-threads INTEGER   If 0, runs on the GPU if available  [default: 0]
    -v, --verbosity             [default: 1]
    -d, --debug                 [default: False]
    -l, --log-dir DIRECTORY     [default: logs]
    -n, --no-log-file           [default: False]
    --help                      Show this message and exit.

Commands:
    data  Base group for data management.
    init  Creates and initializes directory.
    run   Base group for running algorithms.

```

</details>

<details>
<summary>tecpg <b>data</b></summary>

```

Base group for data management.

Options:
    --help  Show this message and exit.

Commands:
    dummy  Generates dummy data.
    gtp    Downloads and extracts GTP data.

```

</details>

<details>
<summary>tecpg data <b>dummy</b></summary>

```

Generates dummy data.

Generates dummy data in the output directory with a given size with file
names M.csv, G.csv, and C.csv.

Options:
    -s, --samples INTEGER
    -m, --meth-rows INTEGER
    -g, --gene-rows INTEGER
    --help                   Show this message and exit.

```

</details>

<details>
<summary>tecpg <b>run</b></summary>

```

Base group for running algorithms.

Options:
    --help  Show this message and exit.

Commands:
    corr  Calculate the pearson correlation coefficient.
    mlr   Calculates the multiple linear regression.

```

</details>

<details>
<summary>tecpg run <b>corr</b></summary>

```

Calculate the pearson correlation coefficient.

Calculate the pearson correlation coefficient with methylation and gene
expression matrices. Optional compute and save chunking to avoid GPU and CPU
memory limits.

Options:
    -c, --chunks INTEGER       [default: 0]
    -s, --save-chunks INTEGER  [default: 0]
    -f, --flatten              [default: False]
    --help                     Show this message and exit.

```

</details>

<details>
<summary>tecpg run <b>mlr</b></summary>

```

    Calculates the multiple linear regression.

    Calculate the multiple linear regression with methylation, gene expression,
    and covariate matrices. Optional chunking to avoid memory limits.

Options:
    -r, --regressions-per-chunk INTEGER  [default: 0]
    -p, --p-thresh FLOAT
    --full-output             [default: False]
    --no-est                  [default: False]
    --no-err                  [default: False]
    --no-t                    [default: False]
    --no-p                    [default: False]
    --help         Show this message and exit.

```

</details>

<br>

# Notes

[Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)

File `tecpg/requirements.txt` created with `pipreqs --mode gt --force`.
