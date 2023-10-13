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

`python dicleave.py --mode --input_path --data_index --output_path`

- **mode**: Designate DiCleave mode, should be "3", "5" or "multi". DiCleave will work on binary classification mode if the value is "3" or "5". DiCleave will work on multiple classification mode if the value is "multi".
- **input_path**: The path of input dataset. Note that the dataset should be a CSV file.
- **data_index**: Columns index of input dataset. This parameter should be a 4-digit number. Each digit means:
  - Dot-bracket secondary structure sequence
  - Cleavage pattern sequence
  - Complemetary sequence
  - Dot-bracket cleavage pattern sequence
- **output_path**: Path where DiCleave store its result.

We provide a simple example to give an intuitive explanation.

The dataset we use in this example is stored in `./example`. In this dataset, the full-length secondary structure sequence, cleavage pattern sequence, complementary sequence and cleavage pattern secondary structure are located in the third column, the fourth column, the sixth column and the fifth column, respectively. Therefore, the `--data_index` parameter should be 2354 (Index of Python starts from 0).

We use the multiple classification mode of DiCleave:

`python dicleave.py --mode multi --input_path ./example/DiCleave_dataset.csv --data_index 2354 --output_path ./example`

<br>
<br>

### Train your DiCleave


We open the API and source code of DiCleave in :page_facing_up: **model.py** and :page_facing_up: **dc.py** files. It can help you to use DiCleave, or to modify and customize your own model based on DiCleave. You can find the API reference [here](https://bic-1.gitbook.io/dicleave/).
