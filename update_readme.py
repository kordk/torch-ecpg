import re

with open('docker-related/README.md', 'r') as f:
    content = f.read()

# Update the running container section
updated_section = """## Running the Container

The container runs the `tecpg` CLI by default. Because it uses `CMD ["python3", "-m", "tecpg"]` without an `ENTRYPOINT`, you must provide the full command if you wish to override the default. You can mount a local directory for input data, intermediate files, and outputs.

A typical invocation on a GPU-enabled host looks like:

```bash
docker run --rm -it --gpus all -v /path/to/host/dir:/work tecpg-pipeline python3 -m tecpg [tecpg_args...]
```

For example, to run the MLR module:
```bash
docker run --rm -it --gpus all -v /path/to/host/dir:/work tecpg-pipeline python3 -m tecpg run mlr ...
```

To run a shell inside the container instead of the `tecpg` CLI:

```bash
docker run --rm -it --gpus all -v /path/to/host/dir:/work tecpg-pipeline bash
```"""

content = re.sub(r'## Running the Container.*?```bash\ndocker run --rm -it --gpus all -v /path/to/host/dir:/work tecpg-pipeline bash\n```', updated_section, content, flags=re.DOTALL)

with open('docker-related/README.md', 'w') as f:
    f.write(content)
