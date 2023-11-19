# Dataset

We upload the training sets and test sets of DiCleave.

Each dataset consists of 7 columns:

* **unnamed**: Used as index.
* **name**: ID of each cleavage pattern entity.
* **sequence**: Full-length pre-miRNA sequence of cleavage pattern.
* **dot_bracket**: Dot-bracket secondary structure of pre-miRNA.
* **cleavage_window**: Sequence of cleavage pattern.
* **window_dot_bracket**: Dot-bracket secondary structure of cleavage pattern.
* **cleavage_window_comp**: Complementary sequence of cleavage pattern.
* **label**: Whether a cleavage window contains Dicer cleavage site. 0 indicates negative pattern; 1 indicates positive pattern from 5' arm; 2 indicates positive pattern from 3' arm.
