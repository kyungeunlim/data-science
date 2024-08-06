# Data Science

Welcome to the `data-science` repository! This repository contains a collection of data science utilities and tools to help streamline your data analysis and machine learning workflows.

## Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The `data-science` repository provides a set of utility functions and modules designed to assist data scientists and analysts in their daily tasks. The primary focus is on making data manipulation, visualization, and analysis more efficient and less error-prone using customized pandas functions.

## Installation

To get started, clone the repository to your local machine:

```sh
git clone https://github.com/kyungeunlim/data-science.git
```

Navigate to the repository directory:

```sh
cd data-science
```

Install the required dependencies:
```sh
pip install -r requirements.txt
```

## Usage

Here are examples of how to use some of the key functions provided in this repository.

### Example Usage of `check_data` Function

```python
import pandas as pd
from utils.analysis_utils import check_data

# Load a sample dataframe
df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': ['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01'],
    'C': ['foo', 'bar', 'foo', 'baz']
})

# Check data
check_data(df)
```

## Modules

### `utils/analysis_utils.py`

This module provides a collection of utility functions for data analysis using Pandas and other scientific computing libraries.


## Contributing

Contributions to this project is welcomed! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

### How to Contribute

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
