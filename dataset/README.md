# Dataset

We upload the training datasets and test datasets of DiCleave here.


Each dataset consists of 7 columns:
* **name**: Name of pre-miRNA.
* **sequence**: Full-length sequence of pre-miRNA.
* **dot_bracket**: Secondary structure of pre-miRNA in dot-bracket format.
* **cleavage_window**: Sequence of a 14 nucleotides long RNA segment (cleavage pattern).
* **window_dot_bracket**: Secondary structure segment of cleavage pattern.
* **cleavage_window_comp**: Complementary sequence of cleavage pattern based on secondary structure.
* **label**: Whether a cleavage window contains Dicer cleavage site. 0 indicates negative pattern; 1 indicates positive pattern from 5' arm; 2 indicates positive pattern from 3' arm
