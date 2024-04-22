# Write & Build the Documentation

This documentation is about how to write and build EvoX's documentation.

## Documentation structure

There are mainly two types of documentation in EvoX: **Guide** and **API**.
**Guide** is for users who want to learn how to use EvoX, mainly written in jupyter notebook format.
**API** is for users/developers who want to know the details of EvoX's code, mainly written as docstrings in the code.

## Multi-language support

EvoX's documentation supports multiple languages. Currently, we support English and Chinese.

### Build the documentation in another language

You will need to install `sphinx-intl`.

```bash
cd docs # the docs folder in the root directory of EvoX
make gettext
sphinx-intl update -p build/gettext -l zh
make -e SPHINXOPTS="-D language='zh'" html
```

### Translate the documentation

The translation is hosted on [Weblate](https://hosted.weblate.org/projects/evox/evox/). You can contribute to the translation there.
