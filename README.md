# DiCleave

DiCleave is a deep neural network model to predict Dicer cleavage site of human precusor microRNAs (pre-miRNAs).


The inputs of DiCleave is a RNA sequence segment and its corresponding pre-miRNA secondary structure embedding. Concretely, the length of RNA sequence segment is 14 nucleotide, with 13 dimensions.


<img src="/img/input.png" alt="input" height="256">

As shown above, the encoding of input RNA sequence is composed of three parts. The yellow part is the encoding of sequence itself, which occupies 5 dimensions (A, C, G, U, O). The blue part is the encoding of complementary sequence, which also occupies 5 dimensions. The symbol "O" indicates unpaired base. Note that "O" is redundant in yellow part. The last three dimensions are designated to the secondary structure of RNA sequence segment, encoded in dot-bracket format.


The pre-miRNA secondary structure embedding is a 64-dimensional vector acquired from an autoencoder.
<br>
<br>
<br>
**To verify the results from our article, clone this repository and run :page_facing_up: evalute.py file**

You need to provide a command line parameter `mode` when runing :page_facing_up: **evalute.py file**. `mode` accepts two values, "binary" and "multi", to evaluate the results of binary models and multiple model, respectively. Thus, first change the working directory to DiCleave, then run `python3 evaluate.py binary` to verify binary models, or run `python3 evaluate.py multi` for multiple model results verification.
<br>
<br>
<br>
The data to verify our model is provided in **dataset**. We also provide the data that we used to train the models. You can merge test sets and training sets to get the raw dataset we employed in this study. In **paras**, we provides well-tuned model parameters for off-the-shelf usage.


We open the API and source code of DiCleave in :page_facing_up: **model.py** and :page_facing_up: **dc.py** files. It can help you to use DiCleave, or to modify and customize your own model based on DiCleave.
