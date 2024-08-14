# CSR5d project

This repository includes scripts for generating results in our 2024 GRL paper, [Deciphering the Role of Total Water Storage Anomalies in Mediating Regional Flooding](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL108126).

## Explanation of scripts

**Data handling**

- `downloadERA5SWE.py`:  Download ERA5 Snow Water Equivalent data from CDS portal
- `downloadGloFAS.py`:  Download GloFAS data from CDS portal
- `parseYangtze.py`:   Parse Yangtze station data into GRDC format

**Analyses**
- `grdc.py`: Main code for CSR.5d analysis. This generates Figures 1, 2,3 and Figure S1, S3 in the paper
- `grdc_monthly5d.py`: Main code for upsampled CSR.monthly analysis (linearly interpolated to the same 5-day intervals as CSR.5d). This generates Figure S2 in the paper

**Utility functions**
- `myutils.py`: Various utility functions
- `dataloader_global.py`, `csr5dloader.py`: data loading and pre-processing functions
- `glofas_all_new.py`: code for extracting GloFAS flow series from GRDC gage locations

### Dependencies

The following is a list of major dependencies. A full list is provided in conda_env.yaml
- `rioxarray`, `geopandas`, `tigramite`, `cartopy`, `xarray`, `shapely`

We used data from GloFAS, ERA5, and GRDC, and Köppen–Geiger. The references are given in the paper.

The CSR.5d dataset is maintained by Dr. Himanshu Save (himanshu.save@csr.utexas.edu). 

If you use materials in this repo, please consider citing our GRL paper

```
@article{sun2024deciphering,
  title={Deciphering the Role of Total Water Storage Anomalies in Mediating Regional Flooding},
  author={Sun, Alexander Y and Save, Himanshu and Rateb, Ashraf and Jiang, Peishi and Scanlon, Bridget R},
  journal={Geophysical Research Letters},
  volume={51},
  number={16},
  pages={e2023GL108126},
  year={2024},
  publisher={Wiley Online Library},
  doi={10.1029/2023GL108126}
}
```

