# Summary Image Outlier Detection Tool

This tool generates a PDF of summary images and a CSV report to help identify image outliers.  

**Intended usage:** Add this tool to your own UV project or Python environment and call its functions with your data. The included notebook is **illustrative only** — it won’t run without access to the example data on Sherlock.

---

## Installation

### Using UV

```bash
uv add https://github.com/jmumford/fmri-outlier-detector.git
```

Then you can import and use it in your UV project:

```python
from fmri_outlier_detector.plotting_functions import generate_all_data_summaries
```

### Using Conda / pip

```bash
pip install git+https://github.com/jmumford/fmri-outlier-detector.git
```

Then import as above.

---

## Usage

```python
generate_all_data_summaries(
    data_dictionaries=my_data,
    n_std=2,
    output_dir="./outlier_results"
)
```

- **data_dictionaries**: List of dicts, each with keys:
  - `main_title` (str)
  - `nifti_paths` (list of str)
  - `image_labels` (list of str)
  - `data_type_label` (str)
- **n_std**: Threshold for outliers (default 2)
- **output_dir**: Where PDFs and CSVs will be saved

---

**Note on the example notebook in example_application:** It demonstrates how the function could be called, but is not runnable outside the original data environment. Users should adapt the function calls to their own data.  There's an example of the output files in `example_application/example_output`
