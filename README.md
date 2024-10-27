# Classification & Deeper Analysis of Mental Health Concerns 

## Objective

Developing a NLP-based solution that automatically extracts, classifies, and performs deeper analysis of mental health concerns from user input

## Directory Structure

Following is the directory structure of the project:

```
.
├── README.md
├── classifiers
│   ├── dataset.csv
│   ├── emotion_classifier.py
│   ├── evaluate_roberta_cat-int.py
│   ├── evaluate_roberta_polarity.py
│   ├── roberta_classifier.py
│   └── roberta_polarity.py
├── datasets
│   ├── dataset.csv
│   └── mental_health_dataset.csv
├── docs
│   ├── Hackathon Task.pdf
│   └── mental_health_concern_classification_using_nlp.pdf
├── generate_data
│   ├── extract.py
│   └── gen_ds.py
├── questionnaire
│   ├── app.py
│   ├── questions.json
│   ├── responses.pkl
│   ├── responses_.pkl
│   └── validate_questions.py
└── timeline-analysis
    ├── emotions_user1.csv
    ├── emotions_user2.csv
    ├── timeline.py
    ├── user1.png
    └── user2.png
```

## Classifiers

Source files are -
1. ``` dataset.csv ``` - This contains the dataset generated.
2. ``` emotion classifier ``` - To extract the emotion from the input sentence.
3. ``` roberta_polarity ``` - Code to finetune classifier RoBERTa to find polarity of given input sentence.
4. ``` roberta_classifier.py ``` - Code to finetune classifier RoBERTa to find intensity and mental concern of given input sentence.
5. ```  evaluate_roberta_polarity.py ``` - Code to evaluate the polarity when user gives input.
6. ``` evaluate_roberta_cat-int.py ``` - Code to evaluate the concern and intensity depending on user input.\

To run any of the files above, use the below command

```
python file_name.py

```

## Link to models

- Finetuned Polarity and Concerned Classifier - Link[https://iiitaphyd-my.sharepoint.com/:f:/g/personal/abhinav_raundhal_students_iiit_ac_in/ElEvnhsnYqtElbLzJJU_5NkBTQ0j6ftTU_tP8tda9-80gw?e=O6DaP0 ]