# dpLGAR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![versions](https://img.shields.io/pypi/pyversions/hydra-core.svg) [![CodeStyle](https://img.shields.io/badge/code%20style-Black-black)]()

The Differentiable Parameter Learning Layered Green & Ampt with Redistribution (dpLGAR) model is a 
differentiable implementation of LGAR (see below). All operations of the model are coded using PyTorch to track gradients
and tune model parameters. 
### Lumped Arid/Semi-arid Model (LASAM) for infiltration and surface runoff

The LASAM simulates infiltration and runoff based on Layered Green & Ampt with redistribution (LGAR) model. 
LGAR is a model which partitions precipitation into infiltration and runoff, 
and is designed for use in arid or semi arid climates. LGAR closely mimics precipitation partitioning results simulated 
by the famous Richards/Richardson equation (RRE), without the inherent reliability and stability challenges the RRE poses. 
Therefore, this model is useful when accurate, stable precipitation partitioning simulations are desired in arid or semi arid areas. 

LGAR has its C version that is available [here](https://github.com/NOAA-OWP/LGAR-C). 

**Published Papers:**
- _Layered Green & Ampt Infiltration with Redistribution_ https://doi.org/10.1029/2022WR033742

BibTeX:
```BibTeX
@article{LaFollette_Ogden_Jan_2023, 
    title={Layered green & AMPT infiltration with redistribution},
    DOI={10.1029/2022wr033742}, 
    journal={Water Resources Research}, 
    author={La Follette, Peter and Ogden, Fred L. and Jan, Ahmad}, 
    year={2023}
} 
```
## Installation

Use conda to create your own env based on our `environment.yml` file
```bash
conda env create -f environment.yml
conda activate lgar
```

## Running this code

We are using [Hydra](https://github.com/facebookresearch/hydra) to store/manage configuration files. 

The main branch code is currently configured to run the Phillipsburg, KS
test case. If you want to use your own case, you will need to manage three config 
files located here:

- `dpLGAR/config.yaml`
  - The main config file. I recommend looking at the Hydra config docs [here](https://hydra.cc/docs/1.3/intro/)
  to learn how this file is structured. 
- `dpLGAR/models/config/base.yaml`
  - This holds all config values for the models
- `dpLGAR/data/config/<site_name>.yaml`
  - This holds all config values for the dataset you're working on.

To run the code, just run the following command inside the `dpLGAR/` folder:

```python
python -m dpLGAR
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
