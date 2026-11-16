# CS 6320 Project 4: Depth Estimation using Stereo

## Setup

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you do not have conda.
2. Create the environment, substituting your OS (`linux`, `mac`, or `win`):
   `conda env create -f proj4_env_<OS>.yml`
3. Activate it: `conda activate proj4`
4. Install the project package from this folder: `pip install -e .`
5. Start the notebook: `jupyter notebook ./proj4_code/simple_stereo.ipynb`
6. Check your work with the `verify(...)` calls in the notebook, or run the whole suite from
   this folder: `pytest unit_tests`
7. Build your submission: `python zip_submission.py --uid <your_uid>`, then upload the zip
   to Gradescope.

## What you implement

| File | Function | Points |
| --- | --- | --- |
| `proj4_code/utils.py` | `generate_random_stereogram()` | 15 |
| `proj4_code/similarity_measures.py` | `ssd_similarity_measure()`, `sad_similarity_measure()` | 15 |
| `proj4_code/disparity_map.py` | `calculate_disparity_map()` | 20 |
| `proj4_code/disparity_map.py` | `calculate_cost_volume()` | 10 |

The remaining 40 points are the sixteen reflection questions in the notebook. Part 4 is worth
10 points of extra credit: one more function plus Questions 17 and 18. The AI-use disclosure at
the end is required and carries no points.


## Notes

- `zip_submission.py` packages `proj4_code/`, `unit_tests/`, `semiglobalmatching/` and `data/`,
  so your code and your run notebook go up together. The submission must stay under 80 MB.
- Use relative paths only. Absolute paths will break on the grader's machine.
- Please do not install extra packages: the graders run this exact environment.
