# Setting up Environment

In case you are facing problem when using DiCleave, we provide two environments.The minimum evironment(**env_minm**) is the environment which DiCleave was build on. The recommended environment(**env_rec**) is an environment which contains dependency packages with much more newer version. These environments are list below:

**env_min**
- python == 3.7.9
- numpy == 1.21.2
- pandas == 1.2.5
- scikit-learn == 1.0.2
- torch == 1.12.1

<br>

**env_rec**
- python == 3.11.3
- numpy == 1.24.3
- pandas == 2.0.3
- scikit-learn == 1.2.2
- torch == 2.0.1

<br>

To build the environment, we recommend you to use Conda. If you haven't installed Conda yet, please check [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for more information.


With Conda installed, you can now build the environment from environment dependencies files. [:page_facing_up] **dc_env_min.yaml** contains the dependencies to create minimum environment, while [:page_facing_up] **dc_env_rec.yaml** contains dependecies to create recommended environment. We show an illustrative example as blow.

In this example, we will build a recommended environment for DiCleave. First, chagne the working directory to DiCleave path:

`cd /<YOUR DIRECTORY>`

<br>

Then, use `conda env create` command to build the environment:

`conda env create --file ./env/dc_env_rec.yaml`

<br>

The default name for recommended envrionment is dc_env_rec, you can find this environment by using:

`conda env list`

<br>

To activate the environment, please run:

`conda activate dc_env_rec`

<br>

Now your environment is ready, you can run DiCleave in this environment successfully.

Finally, when you finish using DiCleave, you can quit this environment by:

`conda deactivate`
