# An Evaluation Framework for Legal Document Summarization

This is the official repository of the paper **An Evaluation Framework for Legal Document Summarization** accepted in the Poster and Demo Session in the [13th Edition of its Language Resources and Evaluation Conference (LREC 2022)](https://lrec2022.lrec-conf.org/en/) 

## Authors - Ankan Mullick†, Abhilash Nandy†, Manav Nitin Kapadnis†, Sohan Patnaik, R Raghav, Roshni Kar

## Abstract 
A law practitioner has to go through numerous lengthy legal case proceedings for their practices of various categories, such as land dispute, corruption, etc. Hence, it is important to summarize these documents, and ensure that summaries contain phrases with intent matching the category of the case. To the best of our knowledge, there is no evaluation metric that evaluates a summary based on its intent. We propose an automated intent-based summarization metric, which shows a better agreement with human evaluation as compared to other automated metrics like BLEU, ROUGE-L etc. in terms of human satisfaction. We also curate a dataset by annotating intent phrases in legal documents, and show a proof of concept as to how this system can be automated.

### Sections
1. Paper
2. Repository Organisation
3. Files Descriptions
4. Demonstration
5. Poster
6. Citation
7. Author Contact Details

## Paper
Our paper can be found [here](toBeAdded).  
<!--Our presentation for the conference can be found [here]()-->

## Repository Organization

```bash
├── JointBert_input_dataset_indian/
|    ├── train/
|          ├── label
|          ├── seq.in
|          └── seq.out
|    ├── dev/
|          ├── label
|          ├── seq.in
|          └── seq.out
|    ├── test/
|          ├── label
|          ├── seq.in
|          └── seq.out
|    ├── intent_label.txt
|    └── slot_label.txt
├── dataset/
|    ├── australian_data/
|          ├── aus_phrases/
|          ├── aus_text/
|          ├── aus_data_statistics.csv
|          └── aus_labels.csv
|    ├── indian_data/
|          ├── ind_phrases_2/
|          ├── ind_text/
|          ├── ind_dataset_statistics.csv
|          └── aus_labels.csv
├── demo_app_code/
├── summarization_models/
|    ├── Abstractive summarization/
|    ├── Bert/
|    ├── Graphical_Model/
|    ├── LetSum Model/
|    ├── Case_Summarizer.ipynb
|    ├── Graphical_Model.ipynb
|    ├── legal-led.ipynb
|    ├── requirements.txt
├── README.md
```

## Files Descriptions

1. **JointBert_input_dataset_indian** - This folder contains the indian dataset formatted to JointBERT input dataset format. Please refer to [JointBERT repository](https://github.com/monologg/JointBERT) for more details on how to convert data to their format.
2. **dataset** - Contains the main indian and australian data with NER labels as well as sentence classification labels. The folder named as "ind_phrases_2" and "aus_phrases" contains the extracted intent phrases (indian were manually annotated, australian were automatically extracted using JointBERT). The files "ind_labels.csv" and "aus_labels.csv" contains the different types of labels used for our experiments.
3. **demo_app_code** - This folder contains the code for the demonstration hosted on the [website](https://share.streamlit.io/manavkapadnis/demolegalevaluation_lrec2022/main/app.py)
4. **summarization_models** -  This folder contains the implementation of different types of summarization models used for our experimentations.


## Demonstration

The code for the demostration can be found in the "demo_app_code/" folder in this repository.<br>
Additionally, the live demo is deployed on this [Demo Link](https://share.streamlit.io/manavkapadnis/demolegalevaluation_lrec2022/main/app.py)<br>
Note: Please wait for 2 minutes, if the website isn't showing the demo website (since the app is dynamically being hosted in livetime when you click on the link).<br>

## Poster
![Poster Image](https://github.com/manavkapadnis/LegalEvaluation_LREC2022/blob/main/LREC_poster_image.PNG)

## Citation
  
  If you want to us this code in your work, please add the following citation -
  
  ```
 To be Added Later
  ```
## Corresponding Author Contact Details

[Manav Nitin Kapadnis](mailto:iammanavk@gmail.com) <br>
[Abhilash Nandy](mailto:nandyabhilash@gmail.com) <br>


**Note**: † - equal contribution
