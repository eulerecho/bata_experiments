## Installation

You first need to download MuJoCO and obtain a licence key from [here](https://www.roboti.us/index.html)

[1] Create conda environment
```
conda create --name bata python=3.7
conda activate bata
```

[2] Install [mujoco_py](https://github.com/openai/mujoco-py) and [gym](https://gym.openai.com/docs/#installation)

[3] Clone the repo
```
cd bata_experiments
conda env update -f setup/environment.yml
```

[4] Install mjrl

```
cd mjrl
pip install -e .
```


[6] Install mjmpc
```
cd mjmpc
pip install -e .
```

## Example
Take a look at the examples directory.
```
cd mjmpc/examples
```

For example, to run MPPI for the rolling task with BATA, run the following
```
python example_mpc.py --config_file configs/pretouch_gripper.yml --controller_type  mppi
```

The parameters for the controller is in `examples/configs` folder.
