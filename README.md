# DiCleave

DiCleave is a deep neural network model to predict Dicer cleavage site of human precusor microRNAs (pre-miRNAs).

We define a cleavage pattern of a pre-miRNA is a 14 nucleotides long sequence segment. If the Dicer cleavage site is located at the center of a cleavage pattern, we label this cleavage site as positive. Accordingly, if a cleavage pattern contains no Dicer cleavage pattern, then we label it as negative.
<br>
<br>
<br>
<img src="/img/cleav_patt.png" alt="cleavage pattern" height="256">

*Illustration of cleavage pattern, example used is predicted secondary structure of hsa-mir548ar*
<br>
<br>
<br>
We illustrate the concept of cleavage pattern and complementary sequence. The red box indicates cleavage pattern at 5' arm. The red asterisk indicate 5' arm Dicer cleavage site. Sequence above is the complementary sequence of this cleavage pattern. Note that the 5th and last two base are unpaired, thus we use symbol "O" to represent this structure.

The inputs of DiCleave is a combination of sequence of cleavage pattern, its complementary sequence and its secondary structure in dot-bracket format. Therefore, the shape of inputs is 14\*13.
<br>
<br>
<br>
<img src="/img/input_.png" alt="input" height="256">

*Input of DiCleave*
<br>
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
<br>

## Usage

To use DiCleave, first clone it to your local repository.

`git clone `

**To verify the results from our article, clone this repository and run :page_facing_up: evalute.py file**

You need to provide a command line parameter `mode` when runing :page_facing_up: **evalute.py file**. `mode` accepts two values, "binary" and "multi", to evaluate the results of binary models and multiple model, respectively. Thus, first change the working directory to DiCleave, then run `python3 evaluate.py binary` to verify binary models, or run `python3 evaluate.py multi` for multiple model results verification.
<br>
<br>
<br>
The data to verify our model is provided in **dataset**. We also provide the data that we used to train the models. You can merge test sets and training sets to get the raw dataset we employed in this study. In **paras**, we provides well-tuned model parameters for off-the-shelf usage.


We open the API and source code of DiCleave in :page_facing_up: **model.py** and :page_facing_up: **dc.py** files. It can help you to use DiCleave, or to modify and customize your own model based on DiCleave. You can find the API reference [here](https://bic-1.gitbook.io/dicleave/).
