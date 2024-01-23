# DeepLearningProject
Deep learning project - Video games sales prediction using Keras and tensorflow

### Dataset can be found here:
https://www.kaggle.com/datasets/sidtwr/videogames-sales-dataset?select=Video_Games_Sales_as_at_22_Dec_2016.csv

# Instructions

### To run this project you need to do the following:

 - Clone the project by downloading it or running the following comand in the terminal: `git clone https://github.com/GrigorAndrei/DeepLearningProject.git`
 - To run the training script you can run the main.py file inside the editor or by running the following command in the editor: `python main.py`

 The main.py file will run a script that installs all the dependencies of the project, a second script that will normalize the data and a third script that will train the models.

**To configure the number of times the model runs through the dataset you need to enter the train_model.py file and modify the value of the variable number_of_passes.**
**To configure the size of the validation set you can change the value of the variable validation_ratio.**

## After running the script

After running the script you will be able to see the training results inside the results folder.
Inside the results folder you will also be able to find the model that gave the best results according to the input data you provided.
Here you will be able to see two plots.
One plot will show the evolution of the accuracy of the models over the training cycles.
The second plot will show the Mean Square Error of each model at the end of the training cycles.
