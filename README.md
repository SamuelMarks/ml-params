ml_params
=========
![Python version range](https://img.shields.io/badge/python-2.7%20|%203.5%20|%203.6%20|%203.7%20|%203.8%20|%203.9rc1-blue.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Linting, testing, and coverage](https://github.com/SamuelMarks/ml-params/workflows/Linting/badge.svg)](https://github.com/SamuelMarks/ml-params/actions)
![Tested OSs, others may work](https://img.shields.io/badge/Tested%20on-Linux%20|%20macOS%20|%20Windows-green)
![Documentation coverage](.github/doccoverage.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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

    $ python -m ml_params --help
    usage: python -m ml_params [-h] [--version] [--engine {tensorflow}]
    
    Consistent CLI for every popular ML framework.
    
    optional arguments:
      -h, --help            show this help message and exit
      --version             show program's version number and exit
      --engine {tensorflow}
                            ML engine, e.g., "TensorFlow", "JAX", "pytorch"
    usage: python -m ml_params [-h] [--version] [--engine {tensorflow}]

Note that this is dynamic, so if you set `--engine` or the `ML_PARAMS_ENGINE` environment variable, you'll get this output:

    $ python -m ml_params --engine 'tensorflow' --help
    usage: python -m ml_params [-h] [--version] [--engine {tensorflow}] {load_data,load_model,train} ...
    
    Consistent CLI for every popular ML framework.
    
    positional arguments:
      {load_data,load_model,train}
                            subcommand to run. Hacked to be chainable.
    
    optional arguments:
      -h, --help            show this help message and exit
      --version             show program's version number and exit
      --engine {tensorflow}
                            ML engine, e.g., "TensorFlow", "JAX", "pytorch"

### TensorFlow example

First let's get some help text:

    $ python -m ml_params --engine 'tensorflow' load_model --help
    usage: python -m ml_params load_model [-h] --model
                                          {DenseNet121,DenseNet169,DenseNet201,EfficientNetB0,EfficientNetB1,EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6,EfficientNetB7,InceptionResNetV2,InceptionV3,MobileNet,MobileNetV2,NASNetLarge,NASNetMobile,ResNet101,ResNet101V2,ResNet152,ResNet152V2,ResNet50,ResNet50V2,Xception}
                                          [--call CALL]
                                          [--model_kwargs MODEL_KWARGS]
    
    Load the model. Takes a model object, or a pipeline that downloads &
    configures before returning a model object.
    
    optional arguments:
      -h, --help            show this help message and exit
      --model {DenseNet121,DenseNet169,DenseNet201,EfficientNetB0,EfficientNetB1,EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6,EfficientNetB7,InceptionResNetV2,InceptionV3,MobileNet,MobileNetV2,NASNetLarge,NASNetMobile,ResNet101,ResNet101V2,ResNet152,ResNet152V2,ResNet50,ResNet50V2,Xception}
                            model object, e.g., a tf.keras.Sequential, tl.Serial,
                            nn.Module instance
      --call CALL           whether to call `model()` even if `len(model_kwargs)
                            == 0`
      --model_kwargs MODEL_KWARGS
                            to be passed into the model. If empty, doesn't call,
                            unless call=True.


Now let's run multiple commands, which behind the scenes constructs a `Trainer` object and calls the relevant methods (subcommands) in the order you reference them:

    $ ML_PARAMS_ENGINE='tensorflow'
    $ python -m ml_params load_data --dataset_name 'mnist' \
                                    --data_type 'infer' \
                          load_model --model 'MobileNet' \
                          train --epochs 3 \
                          --loss 'CategoricalCrossentropy' \
                          --optimizer 'Adam' \
                          --output_type 'numpy' 
                          --validation_split '0.5' 
                          --batch_size 128

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
