# text-decoder

## Environment 

To create your conda environment
```bash
conda env create -f environment.yml
```

To export your conda environment (remeber to do this if you change the packages)
```bash
conda env export | grep -v "^prefix: " > environment.yml
```

## Coding Style

### Standard

- We will be using [PEP8](https://peps.python.org/pep-0008/)
- Use copious [type hinting](https://docs.python.org/3/library/typing.html)

### Linting

If you notice that your file doesn't commit the first time, this is because it failed the linting or formating step. Re-stage your files and commit them again, if that fails edit your code to pass the coding guidelines.

### Git

- Push to branches
- At least 2 code reviewers before merge
