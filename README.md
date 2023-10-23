# DiCleave

DiCleave is a deep neural network model to predict Dicer cleavage site of human precusor microRNAs (pre-miRNAs).

We define a cleavage pattern of a pre-miRNA is a 14 nucleotides long sequence segment. If the Dicer cleavage site is located at the center of a cleavage pattern, we label this cleavage site as positive. Accordingly, if a cleavage pattern contains no Dicer cleavage pattern, then we label it as negative.

<br>
<br>

<img src="/img/cleav_patt.png" alt="cleavage pattern" height="256">

*Illustration of cleavage pattern, example used is predicted secondary structure of hsa-mir548ar*

<br>
<br>

We illustrate the concept of cleavage pattern and complementary sequence. The red box indicates cleavage pattern at 5' arm. The red asterisk indicate 5' arm Dicer cleavage site. Sequence above is the complementary sequence of this cleavage pattern. Note that the 5th and last two base are unpaired, thus we use symbol "O" to represent this structure.

The inputs of DiCleave is a combination of sequence of cleavage pattern, its complementary sequence and its secondary structure in dot-bracket format. Therefore, the shape of inputs is 14\*13.

<br>
<br>

<img src="/img/input_.png" alt="input" height="256">

*Input of DiCleave*

<br>
<br>

As shown above, the encoding of input RNA sequence is composed of three parts. The yellow part is the encoding of sequence itself, which occupies 5 dimensions (A, C, G, U, O). The blue part is the encoding of complementary sequence, which also occupies 5 dimensions. The symbol "O" indicates unpaired base. Note that "O" is redundant in yellow part. The last three dimensions are designated to the secondary structure of RNA sequence segment, encoded in dot-bracket format.

Additionally, the secondary structure embedding of pre-miRNA is a 64-dimensional vector, which is acquired from an autoencoder.

<br>
<br>

## Requirement

DiCleave is built with `Python 3.7.9`. It also requires following dependency:
* `PyTorch >= 1.11.0`
* `Pandas >= 1.2.5`
* `Numpy >= 1.21.0`
* `scikit-learn >= 1.0.2`

<br>

The dependency should not be a problem in most of cases because the project has been tested under higher version of these packages. If you do face the dependecy problem, please execute the following code:

`pip install -r requirements.txt`

If you still have any question about dependency, please contact me without hesitation.

<br>
<br>

## Usage

### Verify results from our article

First, clone DiCleave to your local repository:

`git clone https://github.com/MGuard0303/DiCleave.git /YOUR/DIRECTORY/`

<br>

You should find that all files of DiCleave have been cloned to your local repository. Then, change the current directory to your local repository.

`cd /YOUR/DIRECTORY`

<br>

You need to provide a command line parameter `mode` when runing :page_facing_up: **evalute.py file**. When verifying the binary classification model, set `mode` to "binary"; When verifying the multiple classification model, set `mode` to "multi".

i.e.

```
# Verify binary model
python evaluate.py binary

# Verify multiple model
python evaluate.py multi
```

<br>

The data to verify our model is provided in `./dataset`. We also provide the data that we used to train the models. You can merge test sets and training sets to get the raw dataset we employed in this study. In `./paras`, we provides well-tuned model parameters for off-the-shelf usage.

<br>
<br>

### Use DiCleave to make prediction

To make prediction with DiCleave, please use :page_facing_up: **dicleave.py**. The syntax is

`python dicleave.py --mode --input_file --data_index --output_file`

<br>

- **--mode / -m**:  **[Required]**  Designate DiCleave mode, should be "3", "5" or "multi". DiCleave will work on binary classification mode if the value is "3" or "5". DiCleave will work on multiple classification mode if the value is "multi".
- **--input_file / -i**:  **[Required]**  The path of input dataset. The dataset should be a CSV file.
- **--data_index / -di**:  **[Required]**  Columns index of input dataset. This parameter should be a 4-digit number. Each digit means:
  - Full-length dot-bracket secondary structure sequence
  - Cleavage pattern sequence
  - Complemetary sequence
  - Dot-bracket cleavage pattern sequence
- **--output_file / -o**:  **[Required]**  Path of output file.
- **--model_path / -mp**:  **[Optional]**  Path of DiCleave model parameter file. DiCleave model parameter is a .pt file.

We provide a simple example to give an intuitive explanation.

The dataset we use in this example is stored in `./example`. In this dataset, the full-length secondary structure sequence, cleavage pattern sequence, complementary sequence and cleavage pattern secondary structure are located in the 3rd column, the 4th column, the 6th column and the 5th column, respectively. Therefore, the `--data_index` parameter should be 2354 (Index of Python starts from 0).

We use the multiple classification mode of DiCleave:

First, change the working directory to DiCleave directory:

`cd /DICLEAVE/DIRECTORY`

<br>

then,

`python dicleave.py --mode multi --input_file ./example/dc_dataset.csv --data_index 2354 --output_file ./example/result.txt`

<br>
<br>

### Train your DiCleave

We also provide a script, :page_facing_up: **dicleave_t.py**, to allow you train your own DiCleave model, rather than using the default model we used in this study. The syntax is

`python dicleave_t.py --mode --input_file --data_index --output_file --valid_ratio --batch_size -- learning_rate --weight_decay --nll_weight --max_epoch -k --tolerance`

<br>

- **--mode / -m**:  **[Required]**  Designate DiCleave model, should be "3", "5" or "multi". DiCleave will work on binary classification mode if "3" or "5" is provided. DiCleave will work on multiple classification mode if "multi" is provided.
- **--input_file / -i**:  **[Required]**  The path of input dataset. The dataset should be a CSV file. Note that for training a binary DiCleave model, the labels of dataset can only contain 0 and 1.
- **--data_index / -di**:  **[Required]**  Columns index of input dataset. This parameter should be a 5-digit number. Each digit means:
  - Full-length dot-bracket secondary structure sequence
  - Cleavage pattern sequence
  - Complementary sequence
  - Dot-bracket cleavage pattern sequence
  - Labels
- **--output_file / -o**:  **[Required]**  The path of directory to stored trained model parameters.
- **--valid_ratio / -vr**:  **[Optional]**  The ratio of valid set in input dataset, default is 0.1.
- **--batch_size / -bs**:  **[Optional]**  Batch size for each mini-batch during training, default is 20.
- **--learning_rate / -lr**:  **[Optional]**  Learning rate of optimizer, default is 0.005.
- **--weight_decay / -wd**:  **[Optional]**  Weight decay parameter of optimizer, default is 0.001.
- **--nll_weight / -nw**:  **[Optional]**  Weight of each class in NLLLoss function. Should be a list with three elements, the first element represents negative label (i.e. label=0).Default is [1.0, 1.0, 1.0].
- **--max_epoch / -me**:  **[Optional]**  Max epoch of training process, default is 75.
- **-k**  **[Optional]**:  **[Optional]**  Top-k models will be outputed after training. Default is 3, meaning the training process will output 3 best models on validation set.
- **--tolerance / -tol**:  **[Optional]**  Tolerance for overfitting, default is 3. The higher the value, it is more likely to overfitting.

Here, we provide two examples for intuitive explanations.

In the first example, we will train a multiple classification model. The dataset we use is the same in Prediction part. The label is in the 7th column, so the `--data_index` will be 23546 (Python index starts from 0).

To train the multiple classification model, change working directory to DiCleave directory:

`cd /DICLEAVE/DIRECTORY`

then

`python dicleave_t.py --mode multi --input_file ./paras/dc_dataset.csv --data_index 23546 --output_file ./paras --nll_weight 0.5 1.0 1.0`

<br>

We use parameter `--nll_weight` to change the weight of each class in this example.

In second example we will train a binary classification model for cleavage pattern from 5' arm. Because the binary dataset is derived from :page_facing_up: **dc_dataset.csv**, the `--data_index` remains the same. The only change here is `--mode`:

`python dicleave_t.py --mode 5 --input_file ./paras/dc_dataset_5p.csv --data_index 23546 --output_file ./paras`

<br>
<br>

We open the API and source code of DiCleave in :page_facing_up: **model.py** and :page_facing_up: **dc.py** files. It can help you to use DiCleave, or to modify and customize your own model based on DiCleave. You can find the API reference [here](https://bic-1.gitbook.io/dicleave/).
