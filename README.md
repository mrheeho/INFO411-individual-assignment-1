# INFO411-individual-assignment-1

Preface:
Banks are often posed with a problem t o whether or not a client is credit worthy. Banks commonly
employ data mining techniques to classify a customer into risk categories such as category A (highest rating)
or category C (lowest rating).
A bank collects data from
past credit assessments. The f ile creditworthiness.csv contains 2500 of such
assessments. Each assessment lists 46 attributes of a customer. The last attribute (the 47 47-th attribute) is the
result of the assessment. Open the file and study its contents. You will notice that the columns are coded by
numeric values. The meaning of these values is defined in the file definitions.txt . For example, a value 3 in
the 47 47-th column means that the customer credit worthiness is rated "C". Any value of attributes not listed in
definitions.txt is "as is".
This poses a "prediction" problem. A machine is to learn from the outcomes of past assess
ments and, once
the machine has been trained, to assess any customer who has not yet been assessed. For example, the value
0 in column 47 indicates that this customer has not yet been assessed.

Purpose of this task:
You are to start with an analysis of th
e general properties of this dataset by using suitable visualization and
clustering techniques (i.e. Such as those introduced during the lectures), and you are to obtain an insight into
the degree of difficulty of this prediction task. Then you are to design and deploy an appropriate supervised
prediction model (i.e. MLP) to obtain a prediction of customer ratings.

Question 1:
Analyse the general properties of the dataset and obtain an insight into the difficulty of the
prediction task.
Create a statistical analysis of the attributes and their values, then list 5 of the most interesting (most
valuable) attributes. Explain the reasons that make these attributes interesting.
Note: A set of R-script files are provided with t his assignment (included in the assignment1.zip file). The
scripts provided will allow you to produce some first results. However, virtually none of the parameters used
in these scripts are suitable for obtaining a good insight into the general properties
of the given dataset.

Hence your task is to modify the scripts such that informative results can be obtained from which
conclusions about the learning problem can be made. Note that finding a good set of parameters is often very
time consuming in data mini ng. An additional challange is to make a correct interpretation of the results.
This is what you need to do: Find a good set of parameters (i.e. Through a trial and error approach), obtain
informative results then offer an interpretation of the results. W rite down your approach to conducting the
experiments, explain your results, and offer a comprehensive interpretation of the results. Do not forget that
you are also to provide an insight into the degree of difficulty of this learning problem (i.e. From th e results
that you obtained, can it be expected that a prediction model will be able to obtain 100% prediction
accuracy?). Always explain your answers.

Question 2:
Deploy a prediction model to predict the credit
worthiness of customers which have not yet been assessed.
The prediction capabilities of the MLP in the lab of “Classification” was very poor. Your task is to:
1. Describe a valid strategy that maximises the accuracy of predicting the credit rating. Expla
in whyyour strategy can be expected to maximise the prediction capabilities.
2. Use your strategy to train MLP(s) then report your results. Give an interpretation of your results.
What is the best classification accuracy (expressed in % of correctly clas sified data) that you can
obtain for data that were not used during training (i.e. The test set)?
