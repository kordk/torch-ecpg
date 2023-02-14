# tecpg

Python eCpG mapper with CLI using pytorch.

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

If you have issues with using `pip` in the command line, try `python -m pip` or `python3 -m pip`.

## CUDA

`tecpg` can calculate on the CPU or on a CUDA enabled GPU device. CUDA devices are generally faster than CPU computations for sufficiently large inputs.

The program will automatically determine whether there is a CUDA enabled device and use it if available. To force calculation on the CPU, set the `--threads` option to a nonzero integer. This will also set the number of CPU threads used.

## Inputs

Methylation values, gene expression values, and covariates are provided in csv or tsv files in the `<working>/data` directory. For methylation and gene expression, columns are for individual samples and each row is for an id. For the covariates, the columns are the type of covariate and the rows are the sample. Annotation files are used for region filtration and are stored in the `<working>/annot`. They use the `bed6` standard and store the positions of the methylation or gene expression ids.

## Output

`tecpg run mlr` without chunking creates one output file named out.csv by default in the output directory. If any chunking is used, methylation chuking, gene expression chunking, or both, potentially multiple files are created in the output directory. They are labeled `{methylation chunk number}-{gene expression chunk number}.csv`.

The rows labels of a csv output file indicate the gene expression id and the methylation id. The columns indicate what regression results that column represents. Methylation related labels are prefixed with `mt_`, and gene expression related labels are prefixed with `gt_`. The four regression results returned are the estimate `est`, the standard error `err`, the Student's T statistic `t`, and the p-value `p`.

## Chunking

If the input is too large, the computational device may run out of memory. Chunking can help prevent this by partitioning the data into chunks that are computed and saved separately. Chunking sacrifices parallelization, and thus speed, for lower memory. Avoid chunking wherever possible for speed.

For `tecpg run mlr`, there are two types of chunking: methylation chunking and gene expression chunking. Gene expression chunking is preferable to methylation chunking if possible, as it sacrifices parallelization less. Chunking should be avoided unless required to conform to memory constraints. Use `tepcg chunks` to estimate the number of gene expression loci per chunk given certain parameters.

## Filtration

You may want to include only certain regression results. There are two ways of filtering the results:

1. P-value filtration - all p-values are computed first. Then, regression results with a p-value above a supplied threshold are excluded from the output. This decreases output size and thus increases speed as saving is an expensive operation.
2. Region filtration - region filtration requires annotation files that dictate the positions of methylation and gene expression ids. Then, regressions are filtered by one of the following methods:
   - Cis: the position of the gene expression id and methylation id is within a supplied window and they lie on the same chromosome.
   - Distal: the position of the gene expression id and methylation id is outside of a supplied window and they lie on the same chromosome.
   - Trans: the gene expression id and methylation id lie on different chromosomes.
   - All: no filtration.

P-value filtration filters results after calculating the regression. Region filtration filters the input before the regression results are computed.

## MLR approximate p-values

The p-values returned by `tecpg run mlr` are approximations of the proper values. This is because it uses the normal distribution CDF as an approximation of the Student's T distribution CDF. This approximation is more accurate for larger degrees of freedom. As the number of degrees of freedom approaches $+\infty$, the CDF of the normal distribution and the Student's T distribution approach. The approximation is done because pytorch does not support the Student CDF and does not have the needed funtions to implement it efficiently.

Accuracy tests:

- For 336 degrees of freedom and test t-statistic of 1.877, the percent error between the normal CDF and Student CDF is 0.04469%.
- For 50 degrees of freedom and test t-statistic of 1.877, the percent error between the normal CDF and Student CDF is 0.30206%.

The user should determine whether this accuracy is suitable for the task and the degrees of freedom.

This image from https://en.wikipedia.org/wiki/Student%27s_t-distribution shows the deviation of the Student's T distribution CDF from the normal CDF represented as $v=+\infty$:

<details open>
<summary> Student T CDF comparison </summary>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Student_t_cdf.svg/325px-Student_t_cdf.svg.png">
</details>

## Documentation

Currently, the README and the `tecpg ... --help` commands serve as documentation. Within the code, the function docstrings provide a lot of information about the function. The extensive type hints give added insight into the purpose of functions.

## Acknowledgements

This work was partially supported by an NIH NCI MERIT award (R37, CA233774, PI: Kober) and Cancer Center Support Grant (P30, CA082103, Co-I: Olshen).

