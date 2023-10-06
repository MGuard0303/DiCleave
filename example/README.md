# Example

We give a brief introduction about the usage of DiCleave. 

First let us check the dataset. This dataset contains 3,824 entities, divided into 3 classes. Class 1 indicates positive pattern from 5' arm; Class 2 indicate positive pattern from 3' arm; Class 0 indicates negative pattern.

This dataset consists of 7 columns:

* **name**: Pre-miRNA ID.
* **sequence**: Full-length sequence of pre-miRNA.
* **dot_bracket**: Secondary structure of pre-miRNA in dot-bracket format.
* **cleavage_window**: Sequence of cleavage pattern.
* **window_dot_bracket**: Secondary structure segment of cleavage pattern.
* **cleavage_window_comp**: Complementary sequence of cleavage pattern based on secondary structure.
* **label**: Label of each entity.

Note that if you want to use trained DiCleave model, then the length of input sequence should be shorter than 200 nucleotides.
