# Machine-Translation
In this problem, we need to perform direct machine translation. Direct machine translation is the oldest approach to MT. The translation is based on large dictionaries and word-by-word translation with some simple grammatical adjustments. A direct translation system is designed for a specific source and target language pair. 
I have created my corpus from the book Angels and Demons written by Dan Brown. <br>
1.	Link for book written in English language is 
http://romenplus.com/wp-content/uploads/2014/02/Dan-Brown-Angels-Demons-orig.pdf <br>
2.	Link for book written in French language is http://hullofromlabry.free.fr/Ebooks/Brown%20Dan%20%5BRobert%20Langdon%5D%20(2000)%20Anges%20et%20d%C3%A9mons%20(Angels%20&%20demons).pdf
I have picked up 15 sentences from both French and English versions of the book. Further, I have left out 5 sentence pairs of French-English for testing. So, I have a set of 5 sentences which is be my test set and a set of 10 sentences which is my dev set. The dev set will be used for both translation problem and to produce evaluation during the development process. Test set will be used to see if the system I have developed generalizes the problem well.<br>
After I have created the dev and test set, I have created a bilingual French-English dictionary for each word in the corpus. For this, I have used Google Cloud Translation API which provides a simple programmatic interface for translating an arbitrary string into any supported language using state-of-the-art Neural Machine Translation.<br>

#### Requirements:
- NLTK (http://www.nltk.org/install.html)
- Numpy (https://www.scipy.org/scipylib/download.html)

### Problem 1
The input to this translation is stored in the "Output" folder with filenames French dev set "Sentence_fr_dev.txt", English transaltion of dev set "Sentence_en_dev.txt", French test set "Sentence_fr_test.txt" and English translation of the test set "Sentence_en_test.txt". The program also has "dictionary.json" file which is bilingual dictionary for each word in your working corpus.

#### Execution:
- Run ngupta16_Problem1.py to get the solution for problem 1

### Problem 2
Problem 2 contains solution to IBM Model 1 implementation. The model is trained and tested on "es-en" corpus and given file structure is used. This program trains the model using the files in "es-en/train" folder, run the tests on files in "dev" and "test" folders. The output of the translation is recorded in the file with ".translate" extension in the same folder.

#### Execution:
Run ngupta16_Problem2.py to get the transalted file which will used to claculate the bleu score.

### Calculate Bleu Score
The performace of translation is calculated using Bleu Score. The python script bleu_score.py is used for this purpose. 
Note: The script requires Python 2 (not Python3) for execution.
To check the blue score for dev set without pos tagging, python bleu_score.py newstest2012.en newstest2012.translated

To check the blue score for dev set with pos tagging, python bleu_score.py newstest2012.en newstest2012_pos.translated

To check To check the blue score for test set without pos tagging, python bleu_score.py newstest2013.en newstest2013.translated

To check the blue score for test set with pos tagging, python bleu_score.py newstest2013.en newstest2013_pos.translated<br>

**Sentence_en_dev.txt:** Contains 10 English sentences from the Book Angels and Demons

**Sentence_en_test.txt:** Contains 5 English sentences from the Book Angels and Demons

**Sentence_fr_dev.txt:** Contains 10 French sentences from the Book Angels and Demons

**Sentence_fr_test.txt:** Contains 5 French sentences from the Book Angels and Demons

**output_dev.txt:** Contains the entire output of dev set

**output_test.txt:** Contains the entire output of test set

**dictionary.json:** Bilingual dictionary for each word in your working corpus
