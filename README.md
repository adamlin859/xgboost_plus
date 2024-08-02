This zip file contains the code and the PBC dataset for the our paper: A LUPI distillation-based approach: Application to predicting
Proximal Junctional Kyphosis 

The /data folder contains the the raw data for the Primary Biliary Cirrhosis (PBC) dataset. Synthetic dataset is generated through code.

The /models folder contains our propose model XGBoost+ and SVM+.

To run any of the experiments, please run the following command:

`python3 main.py {experiments}`

Options for {experiments}:
- experiment1
- experiment2
- experiment3

#### Experiment 1: Synthetic dataset experiment where |J| = 3 and |H| = 2. This experiment produces Table 1 in our paper.

#### Experiment 2: Synthetic dataset experiment where the we varied the values of |H| and |J|. This experiment produces Appendix Table 1 in our paper. 
This experiment take longer than other experiments.

#### Experiment 3: PBC dataset experiment where the we varied the values of |H| and |J|. This experiment produces Appendix Table 1 in our paper.


The command will run the specified model using the specified dataset over 100 independent trials and produce the
indivdual trial results and averaged results as well. 

To recreate the same enviornment please use the following code with requirements.txt

python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
