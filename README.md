# Example script for Attention Processor

- Clone this repository

```shell
git clone https://github.com/shunk031/diffusers-attention-processor-example /path/to/your/dir
cd /path/to/your/dir
```

- Create a virtual environment with pyenv/pyenv-virtualenv and activate it

```shell
pyenv virtualenv 3.10.11 diffusers-attention-processor-example
pyenv local diffusers-attention-processor-example
```

- Install dependencies with poetry

```shell
pip install -U pip setuptools wheel poetry
poetry install
```

- Run the example script

```shell
poetry run python main.py
```

## Acknowledgements

- diffusers で Attention の処理をカスタマイズする方法 | AttnProcessor https://zenn.dev/prgckwb/articles/4510b3a06b8163 
