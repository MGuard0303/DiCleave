The data we used in our study is provided here. We upload dataset in the form that we trained and tested DiCleave. From the perspective of convenience, we split negative sequences from 5' arm and 3' arm into different sheet.


Each sheet consists of 7 columns:
* **name**: name of pre-miRNA
* **sequence**: sequence of pre-miRNA
* **dot_bracket**: secondary structure of pre-miRNA in dot-bracket format
* **cleavage_window**: sequence of a 14 nucleotides long RNA segment (cleavage window)
* **window_dot_bracket**: secondary structure segment of cleavage window
* **cleavage_window_comp**: complementary sequence of cleavage window based on secondary structure
* **label**: whether a cleavage window contains Dicer cleavage site

Note that in multiple dataset, label 1 indicates a positive cleavage window from 5' arm; label 2 indicates a positive cleavage window from 3' arm; label 0 indicate a negative cleavage, no matter which arm it comes from. 
