# DiCleave

DiCleave is a deep neural network model to predict Dicer cleavage site of human precusor microRNAs (pre-miRNAs).

We define a cleavage pattern of pre-miRNA is a 14 nucleotides long sequence segment. If the Dicer cleavage site is located at the center of a cleavage pattern, we label this cleavage site as positive. Accordingly, if a cleavage pattern contains no Dicer cleavage pattern, then we label it as negative.

<br>
<br>

<img src="/img/cleav_patt.png" alt="cleavage pattern" height="256">

*Illustration of cleavage pattern, example used is predicted secondary structure of hsa-mir548ar*

<br>
<br>

We illustrate the concept of cleavage pattern and complementary sequence. The red box indicates cleavage pattern at 5' arm. The red asterisk indicate 5' arm Dicer cleavage site. Sequence above is the complementary sequence of this cleavage pattern. Note that the 5th and last two bases are unpaired, thus we use symbol "O" to represent this structure.

The inputs of DiCleave is a combination of sequence of cleavage pattern, its complementary sequence and its secondary structure in dot-bracket format. Therefore, the shape of inputs is 14\*13.

<br>
<br>

<img src="/img/input_.png" alt="input" height="256">

*Input of DiCleave*

<br>
<br>

As shown above, the encoding of input is composed of three parts. The yellow part is the encoding of cleavage pattern itself, which occupies 5 dimensions (A, C, G, U, O). The blue part is the encoding of complementary sequence, which also occupies 5 dimensions. The symbol "O" indicates unpaired base. Note that "O" is redundant in cleavage pattern encoding (yellow part). The last three dimensions are designated to the secondary structure of cleavage pattern, encoded in dot-bracket format.

Additionally, the secondary structure embedding of pre-miRNA is a 64-dimensional vector, which is acquired from an autoencoder.

<br>
<br>

## Requirement

DiCleave is built with `Python 3.7.9`. It also requires following dependency:
* `Numpy >= 1.21.0`
* `Pandas >= 1.2.5`
* `scikit-learn >= 1.0.2`
* `PyTorch >= 1.11.0`

<br>

Any environment with the dependecy package version higher than the minimum version should work well. If you have problem when runing DiCleave, we provide environment files to help you set up the proper environment. Please check [here](https://github.com/MGuard0303/DiCleave/tree/main/env) for more information. 

If you still have any question about environment dependency, please contact me without hesitation.

<br>
<br>

## Usage

### Verify results from our article

First, clone DiCleave to your local repository:

`git clone https://github.com/MGuard0303/DiCleave.git /<YOUR DIRECTORY>`

<br>

Besides, if you don't have git installed, or you don't want git to track your local files, it is easy to download all files from GitHub. See this [page](https://docs.github.com/en/repositories/working-with-files/using-files/downloading-source-code-archives) for more information.

You should find that all files of DiCleave have been cloned to your local repository. Then, change the current directory to your local repository.

`cd /<YOUR DIRECTORY>`

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
- **--data_index / -di**:  **[Required]**  Columns index of input dataset. This parameter should be a 4-digit number. Each digit indicates:
  - Full-length dot-bracket secondary structure sequence
  - Cleavage pattern sequence
  - Complemetary sequence
  - Dot-bracket cleavage pattern sequence
- **--output_file / -o**:  **[Required]**  Path of output file.
- **--model_path / -mp**:  **[Optional]**  Path of DiCleave model parameter file. DiCleave model parameter is a .pt file.

<br>

We provide a simple example to give an intuitive explanation.

The dataset we use in this example is stored in `./example`. This dataset consists of 7 columns. The description of this dataset is shown below.
- **unnamed**: Used as index.
- **name**: ID of each cleavage pattern entity.
- **sequence**: Full-length pre-miRNA sequence of cleavage pattern.
- **dot_bracket**: Dot-bracket secondary structure of pre-miRNA.
- **cleavage_window**: Sequence of cleavage pattern.
- **window_dot_cleavage**: Dot-bracket secondary structure of cleavage pattern.
- **cleavage_window_comp**: Complementary sequence of cleavage pattern.
- **label**: Label indicates whether a cleavage pattern contains a cleavage site in its middle.

<br>

As we can see, the full-length secondary structure sequence, cleavage pattern sequence, complementary sequence and cleavage pattern secondary structure are located in the 4th, 5th, 7th and 6th column, respectively. Therefore, the `--data_index` parameter should be 3465 (Index of Python starts from 0).

We use the multiple classification mode of DiCleave:

First, change the working directory to DiCleave directory:

`cd /<YOUR DIRECTORY>`

<br>

then run:

`python dicleave.py --mode multi --input_file ./example/dc_dataset.csv --data_index 3465 --output_file ./example/result.txt`

<br>

To make prediction using DiCleave binary mode, run:

```
# Predict 5' cleavage pattern
python dicleave.py --mode 5 --input_file ./example/dc_dataset.csv --data_index 3465 --output_file ./example/result.txt

# Predict 3' cleavage pattern
python dicleave.py --mode 3 --input_file ./example/dc_dataset.csv --data_index 3465 --output_file ./example/result.txt
```

<br>
<br>

### Train your DiCleave

We also provide a script, :page_facing_up: **dicleave_t.py**, to allow you train your own DiCleave model, rather than using the default model we used in this study. The syntax is

`python dicleave_t.py --mode --input_file --data_index --output_file --valid_ratio --batch_size --learning_rate --weight_decay --nll_weight --max_epoch -k --tolerance`

<br>

- **--mode / -m**:  **[Required]**  Designate DiCleave model, should be "3", "5" or "multi". DiCleave will work on binary classification mode if "3" or "5" is provided. DiCleave will work on multiple classification mode if "multi" is provided.
- **--input_file / -i**:  **[Required]**  The path of input dataset. The dataset should be a CSV file. Note that for training a binary DiCleave model, the label of dataset can only contain 0 and 1.
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
- **--tolerance / -t**:  **[Optional]**  Tolerance for overfitting, default is 3. The higher the value, it is more likely to overfitting.

<br>

Here, we provide two examples for intuitive explanations.

In the first example, we will train a multiple classification model. The dataset we use is the same in Prediction part. The label is in the 8th column, so the `--data_index` will be 34657 (Python index starts from 0).

To train the multiple classification model, change working directory to DiCleave directory:

`cd /<YOUR DIRECTORY>`

<br>

then run:

`python dicleave_t.py --mode multi --input_file ./example/dc_dataset.csv --data_index 34657 --output_file ./example --nll_weight 0.5 1.0 1.0`

<br>

We use parameter `--nll_weight` to change the weight of each class in this example.

In second example we will train a binary classification model for cleavage pattern from 5' arm. Because the binary dataset is derived from :page_facing_up: **dc_dataset.csv**, the `--data_index` remains the same. The only change here is `--mode`:

`python dicleave_t.py --mode 5 --input_file ./example/dc_dataset_5p.csv --data_index 34657 --output_file ./example`

<br>

Similarly, when training DiCleave for 3' cleavage pattern prediction, run:

`python dicleave_t.py --mode 3 --input_file ./example/dc_dataset_3p.csv --data_index 34657 --output_file ./example`

<br>

To make prediction with your own DiCleave model, you should use both :page_facing_up: **dicleave_t.py** and :page_facing_up: **dicleave.py**. For example:

```
# Train model
python dicleave_t.py --output_file <MODEL_PATH>

# Evaluate model
python dicleave.py --model_path <MODEL_PATH>/model_1.pt
```

<br>
<br>

We open the API and source code of DiCleave in :page_facing_up: **model.py** and :page_facing_up: **dc.py** files. It can help you to use DiCleave, or to modify and customize your own model based on DiCleave. You can find the API reference [here](https://bic-1.gitbook.io/dicleave/).
