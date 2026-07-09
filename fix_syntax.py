with open('tecpg/cli.py', 'r') as f:
    lines = f.readlines()

with open('tecpg/cli.py', 'w') as f:
    for line in lines:
        if "'qr_permute' runs permutation-null testing.\"" in line and not line.strip().startswith('"'):
            line = line.replace("'qr_permute'", "\"        'qr_permute'")
        f.write(line)
