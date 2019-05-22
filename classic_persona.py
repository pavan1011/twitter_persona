#
# * File:    classic_persona.py
# *
# * Author1:  Pavan Kumar K N (pavankumar.karkekopp@ucalgary.ca)
# * Date:     20th May 2019
# * Summary of File:
# * Explore mbti_1.csv file acquired from https://www.kaggle.com/datasnaek/mbti-type
# * Apply state-of-the-art reported publicly
# * Build classifier model that is better using machine learning techniques
#

import numpy as np
import pandas as pd
import sklearn

mbti_data = pd.read_csv("data/mbti_1.csv")


