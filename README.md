ml_params
=========
![Python version range](https://img.shields.io/badge/python-2.7%20|%203.5%20|%203.6%20|%203.7%20|%203.8%20|%203.9b5-blue.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Linting, testing, and coverage](https://github.com/SamuelMarks/ml-params/workflows/Linting/badge.svg)](https://github.com/SamuelMarks/ml-params/actions)
![Documentation coverage](.github/doccoverage.svg)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Consistent CLI and Python SDK API for †every popular ML framework.

†that's the goal anyway! - PR or just suggestions for other ML frameworks to add are welcome :grin:

The approach is type-focussed, with explicit static code-generation of `Literal`ally:

  - Transfer learning models
  - Optimizers
  - Losses
  - &etc., including non-NN related scikit.learn, XGBoost

For example, the following would be exposed, and thereby become useful from Python, in GUIs, REST/RPC APIs, and CLIs:
```python
from typing import Literal

losses = Literal['BinaryCrossentropy', 'CategoricalCrossentropy', 'CategoricalHinge', 'CosineSimilarity', 'Hinge',
                 'Huber', 'KLD', 'KLDivergence', 'LogCosh', 'MAE', 'MAPE', 'MSE', 'MSLE', 'MeanAbsoluteError',
                 'MeanAbsolutePercentageError', 'MeanSquaredError', 'MeanSquaredLogarithmicError', 'Poisson',
                 'Reduction', 'SparseCategoricalCrossentropy', 'SquaredHinge']
```

## Developer guide

The [doctrans](https://github.com/SamuelMarks/doctrans) project was developed to make ml-params—and its implementations—possible… without a ridiculous amount of hand-written duplication. The duplication is still present, but doctrans will automatically keep them in sync, multi-directionally. So you can edit any of these and it'll translate the changes until every 'interface' is equivalent:

  - CLI
  - Class
  - Function/method

It will also expand a specific property, like your `get_losses` function could generate the aforementioned `Literal`.

## Install dependencies

    pip install -r requirements.txt

## Install package

    pip install .

## Usage

    python -m ml_params --help

## Official implementations

| Google | Other vendors |
| -------| ------------- |
| [tensorflow](https://github.com/SamuelMarks/ml-params-tensorflow)  | [pytorch](https://github.com/SamuelMarks/ml-params-pytorch) |
| [keras](https://github.com/SamuelMarks/ml-params-keras)  | [skorch](https://github.com/SamuelMarks/ml-params-skorch) |
| [flax](https://github.com/SamuelMarks/ml-params-flax) | [sklearn](https://github.com/SamuelMarks/ml-params-sklearn) |
| [trax](https://github.com/SamuelMarks/ml-params-trax) | [xgboost](https://github.com/SamuelMarks/ml-params-xgboost) |
| [jax](https://github.com/SamuelMarks/ml-params-jax) | [cntk](https://github.com/SamuelMarks/ml-params-cntk) |
 
## Related official projects

  - [ml-prepare](https://github.com/SamuelMarks/ml-prepare)

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
