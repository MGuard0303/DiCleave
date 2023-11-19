# Example

We give a brief introduction about the usage of DiCleave. 

First let us check the dataset. This dataset contains 3,824 entities, divided into 3 classes. Class 1 indicates positive pattern from 5' arm; Class 2 indicates positive pattern from 3' arm; Class 0 indicates negative pattern.

This dataset consists of 8 columns:

* **unnamed**: Used as index.
* **name**:ID of each cleavage pattern entity.
* **sequence**: Full-length pre-miRNA sequence of cleavage pattern.
* **dot_bracket**: Dot-bracket secondary structure of pre-miRNA.
* **cleavage_window**: Sequence of cleavage pattern.
* **window_dot_bracket**: Dot-bracket secondary structure of cleavage pattern.
* **cleavage_window_comp**: Complementary sequence of cleavage pattern.
* **label**: Label indicates whether a cleavage pattern contains a cleavage site in its middle.

<br>

Note that if you want to use trained DiCleave model, then the length of input sequence should be shorter than 200 nucleotides.
