# cnnproject

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

COMP 309 Project for VUW

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         cnnproject and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── cnnproject   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes cnnproject a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


# Program setup for tutors: PLEASE READ

## ENVIROMENT AND INSTALLATION:

- Run 'make create_environment' to setup the initial conda env
- Activate the conda environment generated
- Run 'make requirements' to install pip dependences. (this can also be done manually, but you MUST run 'pip -e .' in order to set the python path correctly).

Install additional requirements via conda, I suggest the following in the following order, however I may have missed one (maybe a default scipi), and therefore I've provided a full list of packages for the venv at the footer of this file in case:

'conda install py-opencv'
'conda install pytorch::pytorch torchvision torchaudio -c pytorch'
'conda install albumentations'
'conda install numpy=1.26' 

This is REQUIRED as the upgrade to numpy 2.0 broke a key dependency for pretty much every image augmentation library out there

## DATA PIPELINE:

The project has custom dataloaders that need to be set up before the testing pipeline can be run.

- Create the directory with the relative path "data/processed/resized"
- Place the full set of test images in "data/processed/resized"

- Run 'dataset.py'. This has been set to create the cutom dataloader needed. You should now see "test_dataset.pkl" in "data/processed"

## ADD TRAINED MODEL:
- The trained model can be downloaded from https://drive.google.com/file/d/1Z78i-P-lX3scQXMIMZsHF8KJ-MnTJVVv/view?usp=drive_link
- Place the model (CNN_epoch_9.pth) in models/
- You can now run 'predict.py' and 'CNN_epoch_9_metrics.txt' should appear in "reports/" THIS MAY TAKE SOME TIME (average ~20mins on my machine)
    
# Full package list for the conda env
packages in environment at /opt/anaconda3/envs/cnnprojectcomp309:

Name                    Version                   Build  Channel
albumentations            1.3.1              pyhd8ed1ab_0    conda-forge
anyio                     4.6.2.post1              pypi_0    pypi
aom                       3.9.1                hf036a51_0    conda-forge
appnope                   0.1.4              pyhd8ed1ab_0    conda-forge
argon2-cffi               23.1.0                   pypi_0    pypi
argon2-cffi-bindings      21.2.0                   pypi_0    pypi
arrow                     1.3.0                    pypi_0    pypi
asttokens                 2.4.1              pyhd8ed1ab_0    conda-forge
async-lru                 2.0.4                    pypi_0    pypi
attrs                     24.2.0                   pypi_0    pypi
babel                     2.16.0                   pypi_0    pypi
beautifulsoup4            4.12.3                   pypi_0    pypi
black                     24.10.0                  pypi_0    pypi
blas                      2.106                       mkl    conda-forge
bleach                    6.1.0                    pypi_0    pypi
blosc                     1.21.6               h7d75f6d_0    conda-forge
brotli                    1.1.0                    pypi_0    pypi
brotli-bin                1.1.0                h00291cd_2    conda-forge
brotli-python             1.1.0           py310h53e7c6a_2    conda-forge
brunsli                   0.1                  h046ec9c_0    conda-forge
bzip2                     1.0.8                hfdf4475_7    conda-forge
c-ares                    1.34.2               h32b1619_0    conda-forge
c-blosc2                  2.15.1               hb9356d3_0    conda-forge
ca-certificates           2024.8.30            h8857fd0_0    conda-forge
cairo                     1.18.0               h37bd5c4_3    conda-forge
certifi                   2024.8.30          pyhd8ed1ab_0    conda-forge
cffi                      1.17.1                   pypi_0    pypi
charls                    2.4.2                he965462_0    conda-forge
charset-normalizer        3.4.0              pyhd8ed1ab_0    conda-forge
click                     8.1.7           unix_pyh707e725_0    conda-forge
cnnproject                0.0.1                    pypi_0    pypi
comm                      0.2.2              pyhd8ed1ab_0    conda-forge
contourpy                 1.3.0                    pypi_0    pypi
cycler                    0.12.1             pyhd8ed1ab_0    conda-forge
dav1d                     1.2.1                h0dc2134_0    conda-forge
debugpy                   1.8.7                    pypi_0    pypi
decorator                 5.1.1              pyhd8ed1ab_0    conda-forge
defusedxml                0.7.1                    pypi_0    pypi
exceptiongroup            1.2.2              pyhd8ed1ab_0    conda-forge
executing                 2.1.0              pyhd8ed1ab_0    conda-forge
expat                     2.6.3                hac325c4_0    conda-forge
fastjsonschema            2.20.0                   pypi_0    pypi
ffmpeg                    7.1.0           gpl_h47bdf92_102    conda-forge
filelock                  3.16.1             pyhd8ed1ab_0    conda-forge
flake8                    7.1.1                    pypi_0    pypi
font-ttf-dejavu-sans-mono 2.37                 hab24e00_0    conda-forge
font-ttf-inconsolata      3.000                h77eed37_0    conda-forge
font-ttf-source-code-pro  2.038                h77eed37_0    conda-forge
font-ttf-ubuntu           0.83                 h77eed37_3    conda-forge
fontconfig                2.14.2               h5bb23bf_0    conda-forge
fonts-conda-ecosystem     1                             0    conda-forge
fonts-conda-forge         1                             0    conda-forge
fonttools                 4.54.1                   pypi_0    pypi
fqdn                      1.5.1                    pypi_0    pypi
freetype                  2.12.1               h60636b9_2    conda-forge
fribidi                   1.0.10               hbcb3906_0    conda-forge
gdk-pixbuf                2.42.12              ha587570_0    conda-forge
geos                      3.13.0               hac325c4_0    conda-forge
giflib                    5.2.2                h10d778d_0    conda-forge
gmp                       6.3.0                hf036a51_2    conda-forge
gmpy2                     2.1.5                    pypi_0    pypi
graphite2                 1.3.13            h73e2aa4_1003    conda-forge
h11                       0.14.0                   pypi_0    pypi
h2                        4.1.0              pyhd8ed1ab_0    conda-forge
harfbuzz                  9.0.0                h098a298_1    conda-forge
hdf5                      1.14.4          nompi_h57e3b00_101    conda-forge
hpack                     4.0.0              pyh9f0ad1d_0    conda-forge
httpcore                  1.0.6                    pypi_0    pypi
httpx                     0.27.2                   pypi_0    pypi
hyperframe                6.0.1              pyhd8ed1ab_0    conda-forge
icu                       75.1                 h120a0e1_0    conda-forge
idna                      3.10               pyhd8ed1ab_0    conda-forge
imagecodecs               2024.9.22                pypi_0    pypi
imageio                   2.36.0             pyh12aca89_1    conda-forge
imath                     3.1.12               h2016aa1_0    conda-forge
imgaug                    0.4.0              pyhd8ed1ab_1    conda-forge
importlib-metadata        8.5.0              pyha770c72_0    conda-forge
ipykernel                 6.29.5             pyh57ce528_0    conda-forge
ipython                   8.29.0             pyh707e725_0    conda-forge
isoduration               20.11.0                  pypi_0    pypi
isort                     5.13.2                   pypi_0    pypi
jasper                    4.2.4                hb10263b_0    conda-forge
jedi                      0.19.1             pyhd8ed1ab_0    conda-forge
jinja2                    3.1.4              pyhd8ed1ab_0    conda-forge
joblib                    1.4.2              pyhd8ed1ab_0    conda-forge
json5                     0.9.25                   pypi_0    pypi
jsonpointer               3.0.0                    pypi_0    pypi
jsonschema                4.23.0                   pypi_0    pypi
jsonschema-specifications 2024.10.1                pypi_0    pypi
jupyter-events            0.10.0                   pypi_0    pypi
jupyter-lsp               2.2.5                    pypi_0    pypi
jupyter-server            2.14.2                   pypi_0    pypi
jupyter-server-terminals  0.5.3                    pypi_0    pypi
jupyter_client            8.6.3              pyhd8ed1ab_0    conda-forge
jupyter_core              5.7.2              pyh31011fe_1    conda-forge
jupyterlab                4.2.5                    pypi_0    pypi
jupyterlab-pygments       0.3.0                    pypi_0    pypi
jupyterlab-server         2.27.3                   pypi_0    pypi
jxrlib                    1.1                  h10d778d_3    conda-forge
kiwisolver                1.4.7                    pypi_0    pypi
krb5                      1.21.3               h37d8d59_0    conda-forge
lame                      3.100             hb7f2c08_1003    conda-forge
lazy-loader               0.4                pyhd8ed1ab_1    conda-forge
lazy_loader               0.4                pyhd8ed1ab_1    conda-forge
lcms2                     2.16                 ha2f27b4_0    conda-forge
lerc                      4.0.0                hb486fe8_0    conda-forge
libabseil                 20240722.0      cxx17_hac325c4_1    conda-forge
libaec                    1.1.3                h73e2aa4_0    conda-forge
libasprintf               0.22.5               hdfe23c8_3    conda-forge
libass                    0.17.3               h5386a9e_0    conda-forge
libavif16                 1.1.1                ha49a9e2_1    conda-forge
libblas                   3.9.0                     6_mkl    conda-forge
libbrotlicommon           1.1.0                h00291cd_2    conda-forge
libbrotlidec              1.1.0                h00291cd_2    conda-forge
libbrotlienc              1.1.0                h00291cd_2    conda-forge
libcblas                  3.9.0                     6_mkl    conda-forge
libcurl                   8.10.1               h58e7537_0    conda-forge
libcxx                    19.1.2               hf95d169_0    conda-forge
libdeflate                1.22                 h00291cd_0    conda-forge
libedit                   3.1.20191231         h0678c8f_2    conda-forge
libev                     4.33                 h10d778d_2    conda-forge
libexpat                  2.6.3                hac325c4_0    conda-forge
libffi                    3.4.2                h0d85af4_5    conda-forge
libgettextpo              0.22.5               hdfe23c8_3    conda-forge
libgfortran               5.0.0           13_2_0_h97931a8_3    conda-forge
libgfortran5              13.2.0               h2873a65_3    conda-forge
libglib                   2.82.2               hb6ef654_0    conda-forge
libhwloc                  2.11.1          default_h456cccd_1000    conda-forge
libhwy                    1.1.0                h7728843_0    conda-forge
libiconv                  1.17                 hd75f5a5_2    conda-forge
libintl                   0.22.5               hdfe23c8_3    conda-forge
libjpeg-turbo             3.0.0                h0dc2134_1    conda-forge
libjxl                    0.11.0               haf62dd7_2    conda-forge
liblapack                 3.9.0                     6_mkl    conda-forge
liblapacke                3.9.0                     6_mkl    conda-forge
libnghttp2                1.64.0               hc7306c3_0    conda-forge
libopencv                 4.10.0          headless_py310h3f01f91_10    conda-forge
libopenvino               2024.4.0             h84cb933_2    conda-forge
libopenvino-auto-batch-plugin 2024.4.0             h92dab7a_2    conda-forge
libopenvino-auto-plugin   2024.4.0             h92dab7a_2    conda-forge
libopenvino-hetero-plugin 2024.4.0             h14156cc_2    conda-forge
libopenvino-intel-cpu-plugin 2024.4.0             h84cb933_2    conda-forge
libopenvino-ir-frontend   2024.4.0             h14156cc_2    conda-forge
libopenvino-onnx-frontend 2024.4.0             he28f95a_2    conda-forge
libopenvino-paddle-frontend 2024.4.0             he28f95a_2    conda-forge
libopenvino-pytorch-frontend 2024.4.0             hc3d39de_2    conda-forge
libopenvino-tensorflow-frontend 2024.4.0             h488aad4_2    conda-forge
libopenvino-tensorflow-lite-frontend 2024.4.0             hc3d39de_2    conda-forge
libopus                   1.3.1                hc929b4f_1    conda-forge
libpng                    1.6.44               h4b8f8c9_0    conda-forge
libprotobuf               5.28.2               h8b30cf6_0    conda-forge
librsvg                   2.58.4               h2682814_0    conda-forge
libsodium                 1.0.20               hfdf4475_0    conda-forge
libsqlite                 3.47.0               h2f8c449_0    conda-forge
libssh2                   1.11.0               hd019ec5_0    conda-forge
libtiff                   4.7.0                h583c2ba_1    conda-forge
libvpx                    1.14.1               hf036a51_0    conda-forge
libwebp-base              1.4.0                h10d778d_0    conda-forge
libxcb                    1.17.0               hf1f96e2_0    conda-forge
libxml2                   2.12.7               heaf3512_4    conda-forge
libzlib                   1.3.1                hd23fc13_2    conda-forge
libzopfli                 1.0.3                h046ec9c_0    conda-forge
llvm-openmp               19.1.2               hf78d878_0    conda-forge
loguru                    0.7.2                    pypi_0    pypi
lz4-c                     1.9.4                hf0c8a7f_0    conda-forge
markdown-it-py            3.0.0              pyhd8ed1ab_0    conda-forge
markupsafe                3.0.2                    pypi_0    pypi
matplotlib                3.9.2                    pypi_0    pypi
matplotlib-base           3.9.2           py310h449bdf7_1    conda-forge
matplotlib-inline         0.1.7              pyhd8ed1ab_0    conda-forge
mccabe                    0.7.0                    pypi_0    pypi
mdurl                     0.1.2              pyhd8ed1ab_0    conda-forge
mistune                   3.0.2                    pypi_0    pypi
mkl                       2020.4             h08c4f10_301    conda-forge
mpc                       1.3.1                h9d8efa1_1    conda-forge
mpfr                      4.2.1                haed47dc_3    conda-forge
mpmath                    1.3.0              pyhd8ed1ab_0    conda-forge
munkres                   1.1.4              pyh9f0ad1d_0    conda-forge
mypy-extensions           1.0.0                    pypi_0    pypi
nbclient                  0.10.0                   pypi_0    pypi
nbconvert                 7.16.4                   pypi_0    pypi
nbformat                  5.10.4                   pypi_0    pypi
ncurses                   6.5                  hf036a51_1    conda-forge
nest-asyncio              1.6.0              pyhd8ed1ab_0    conda-forge
networkx                  3.4.2              pyhd8ed1ab_0    conda-forge
notebook                  7.2.2                    pypi_0    pypi
notebook-shim             0.2.4                    pypi_0    pypi
numpy                     1.26.4                   pypi_0    pypi
opencv                    4.10.0          headless_py310hf0d8d4c_10    conda-forge
openexr                   3.3.1                hb646618_2    conda-forge
openh264                  2.4.1                h73e2aa4_0    conda-forge
openjpeg                  2.5.2                h7310d3a_0    conda-forge
openssl                   3.3.2                hd23fc13_0    conda-forge
overrides                 7.7.0                    pypi_0    pypi
packaging                 24.1               pyhd8ed1ab_0    conda-forge
pandas                    2.2.3                    pypi_0    pypi
pandocfilters             1.5.1                    pypi_0    pypi
pango                     1.54.0               h115fe74_2    conda-forge
parso                     0.8.4              pyhd8ed1ab_0    conda-forge
pathspec                  0.12.1                   pypi_0    pypi
pcre2                     10.44                h7634a1b_2    conda-forge
pexpect                   4.9.0              pyhd8ed1ab_0    conda-forge
pickleshare               0.7.5                   py_1003    conda-forge
pillow                    11.0.0                   pypi_0    pypi
pip                       24.3.1                   pypi_0    pypi
pixman                    0.43.4               h73e2aa4_0    conda-forge
platformdirs              4.3.6              pyhd8ed1ab_0    conda-forge
prometheus-client         0.21.0                   pypi_0    pypi
prompt-toolkit            3.0.48             pyha770c72_0    conda-forge
psutil                    6.1.0                    pypi_0    pypi
pthread-stubs             0.4               h00291cd_1002    conda-forge
ptyprocess                0.7.0              pyhd3deb0d_0    conda-forge
pugixml                   1.14                 he965462_0    conda-forge
pure_eval                 0.2.3              pyhd8ed1ab_0    conda-forge
py-opencv                 4.10.0          headless_py310hc32ca21_10    conda-forge
pycodestyle               2.12.1                   pypi_0    pypi
pycparser                 2.22               pyhd8ed1ab_0    conda-forge
pyflakes                  3.2.0                    pypi_0    pypi
pygments                  2.18.0             pyhd8ed1ab_0    conda-forge
pyparsing                 3.2.0              pyhd8ed1ab_1    conda-forge
pysocks                   1.7.1              pyha2e5f31_6    conda-forge
python                    3.10.15         hd8744da_2_cpython    conda-forge
python-dateutil           2.9.0              pyhd8ed1ab_0    conda-forge
python-dotenv             1.0.1                    pypi_0    pypi
python-json-logger        2.0.7                    pypi_0    pypi
python_abi                3.10                    5_cp310    conda-forge
pytorch                   2.2.2                  py3.10_0    pytorch
pytz                      2024.2                   pypi_0    pypi
pywavelets                1.7.0                    pypi_0    pypi
pyyaml                    6.0.2                    pypi_0    pypi
pyzmq                     26.2.0                   pypi_0    pypi
qhull                     2020.2               h3c5361c_5    conda-forge
qudida                    0.0.4              pyhd8ed1ab_0    conda-forge
rav1e                     0.6.6                h7205ca4_2    conda-forge
readline                  8.2                  h9e318b2_1    conda-forge
referencing               0.35.1                   pypi_0    pypi
requests                  2.32.3             pyhd8ed1ab_0    conda-forge
rfc3339-validator         0.1.4                    pypi_0    pypi
rfc3986-validator         0.1.1                    pypi_0    pypi
rich                      13.9.3             pyhd8ed1ab_0    conda-forge
rpds-py                   0.20.0                   pypi_0    pypi
scikit-image              0.24.0                   pypi_0    pypi
scikit-learn              1.5.2                    pypi_0    pypi
scipy                     1.14.1                   pypi_0    pypi
send2trash                1.8.3                    pypi_0    pypi
setuptools                75.1.0             pyhd8ed1ab_0    conda-forge
shapely                   2.0.6                    pypi_0    pypi
shellingham               1.5.4              pyhd8ed1ab_0    conda-forge
six                       1.16.0             pyh6c4a22f_0    conda-forge
snappy                    1.2.1                he1e6707_0    conda-forge
sniffio                   1.3.1                    pypi_0    pypi
soupsieve                 2.6                      pypi_0    pypi
stack_data                0.6.2              pyhd8ed1ab_0    conda-forge
svt-av1                   2.2.1                hac325c4_0    conda-forge
sympy                     1.13.3          pypyh2585a3b_103    conda-forge
tbb                       2021.13.0            h37c8870_0    conda-forge
terminado                 0.18.1                   pypi_0    pypi
threadpoolctl             3.5.0              pyhc1e730c_0    conda-forge
tifffile                  2024.9.20          pyhd8ed1ab_0    conda-forge
tinycss2                  1.4.0                    pypi_0    pypi
tk                        8.6.13               h1abcd95_1    conda-forge
tomli                     2.0.2                    pypi_0    pypi
torch                     2.2.2                    pypi_0    pypi
torchaudio                2.2.2                    pypi_0    pypi
torchvision               0.17.2                   pypi_0    pypi
tornado                   6.4.1                    pypi_0    pypi
tqdm                      4.66.5                   pypi_0    pypi
traitlets                 5.14.3             pyhd8ed1ab_0    conda-forge
typer                     0.12.5             pyhd8ed1ab_0    conda-forge
typer-slim                0.12.5             pyhd8ed1ab_0    conda-forge
typer-slim-standard       0.12.5               hd8ed1ab_0    conda-forge
types-python-dateutil     2.9.0.20241003           pypi_0    pypi
typing-extensions         4.12.2               hd8ed1ab_0    conda-forge
typing_extensions         4.12.2             pyha770c72_0    conda-forge
tzdata                    2024.2                   pypi_0    pypi
unicodedata2              15.1.0                   pypi_0    pypi
uri-template              1.3.0                    pypi_0    pypi
urllib3                   2.2.3              pyhd8ed1ab_0    conda-forge
wcwidth                   0.2.13             pyhd8ed1ab_0    conda-forge
webcolors                 24.8.0                   pypi_0    pypi
webencodings              0.5.1                    pypi_0    pypi
websocket-client          1.8.0                    pypi_0    pypi
wheel                     0.44.0             pyhd8ed1ab_0    conda-forge
x264                      1!164.3095           h775f41a_2    conda-forge
x265                      3.5                  hbb4e6a2_3    conda-forge
xorg-libxau               1.0.11               h00291cd_1    conda-forge
xorg-libxdmcp             1.1.5                h00291cd_0    conda-forge
xz                        5.2.6                h775f41a_0    conda-forge
yaml                      0.2.5                h0d85af4_2    conda-forge
zeromq                    4.3.5                he4ceba3_6    conda-forge
zfp                       1.0.1                h469392a_2    conda-forge
zipp                      3.20.2             pyhd8ed1ab_0    conda-forge
zlib                      1.3.1                hd23fc13_2    conda-forge
zlib-ng                   2.2.2                hac325c4_0    conda-forge
zstandard                 0.23.0                   pypi_0    pypi
zstd                      1.5.6                h915ae27_0    conda-forge


