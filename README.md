# tecpg

Python eCpG mapper with CLI using pytorch

## Dependencies

Stored in tecpg/requirements.txt:

```
click>=8.0.3
colorama>=0.4.4
matplotlib>=3.5.1
numpy>=1.22.4
pandas>=1.3.5
requests>=2.26.0
scipy>=1.7.2
torch>=1.11.0+cu113
```

## Conditions

Pytorch will run on the CPU or CUDA-supported GPU. A GPU is generally much faster. The device is automatically selected based on `torch.cuda.is_available()`.

Python 3.10 or higher is required.

## Installation

If you would like to use the CLI, run:
`python3 -m tecpg [options] [command]...`

If you want to run tecpg in a python file, tecpg will have to be in your system path.

```py
import tecpg

ecpg.main.main()
```
