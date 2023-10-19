# CPII_Emmy_Noether

I just copied the README-file from the StatPhys2022 course, and changed it a bit. You will probably recognize this once you have your first tutorial in that course.

Humboldt Universität zu Berlin - Computational Physics II WS2023/2024 (Prof. Dr. Agostino Patella)

## Getting started

### 0. Clone and update this repository

To get all data included in this repository, open a terminal, navigate to the directory you want to work in and execute
```
git clone https://github.com/idawoestheinrich/CPII_Emmy_Noether.git

```

This will create a directory `CPII_Emmy_Noether` in the current directory. \
You can do `cd CPII_Emmy_Noether` to change to that directory. If you do `pwd`, you should get the current directory: ...\CPII_Emmy_Noether. \
Any updates that will be posted can be downloaded by running
```
git pull
```

For the Linux/OSX users, it is best to use the git command lines to submit your code.

Change directories to the \submissions folder

You can do it in this order:
- **git pull** to pull any new changes from the repo.
- **git add (file name here)** to add a file change to be committed. Make sure you include the whole file name, including the .ipynb bit for notebooks.
- **git commit -m "message here"** to commit a change. Please enter a simple message here to describe what you're doing.
- **git push** to push the commit to the repo.

For Windows users (and Unix users who do not like command lines...), see Desktop Github: https://desktop.github.com.

### 1. Installing `Python` / `conda`
`Python` is the programming language we will be using. If you're not familiar with `Python` basics there are plenty 
of introductions, see e.g. https://www.learnpython.org.

If you do not have a `Python` instance set up on your computer, we recommend installing `conda`. `conda` does not only 
provide the interpreter for `Python` but is also a package manager, that allows to easily install ad-on packages that 
we will need. You can find more info here: https://docs.conda.io/projects/conda/en/latest/index.html \
We recommend installing the lightweight `conda` called `Miniconda`: https://docs.conda.io/en/latest/miniconda.html

```


### 2. Installing packages separately

You can install all packages via `pip`. The packages you can start with are `numpy`, `matplotlib`, `jupyterlab`, `scipy`, `pandas`, and `astropy`.
Make sure your environment is activated and execute
```
pip install <package name here>
```


### 3. Running `jupyter`

The `jupyter notebook` is an interface to the interactive `python` called `IPython`. 
Notebook files have the file extension `.ipynb`. To start it, activate your environment (if you haven't already) and run
```
jupyter notebook
```

This opens the `jupyter notebook` webpage in your browser. You can now browse for a Notebook file in the file browser on the left
and open it via double click.
