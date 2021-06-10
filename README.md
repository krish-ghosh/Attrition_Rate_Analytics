# Attrition Rate Analytics

<p align="center">
<kbd><img width="500" height="350" src="https://www.datocms-assets.com/17507/1606822945-customer-attrition0.png"></kbd>
</p>

## Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [License](#license)

## Overview

Customer Attrition is a tendency of customers to abandon a brand and stop being a paying client of a particular business. The percentage of customers that discontinue using a company’s products or services during a particular time period is called Customer Attrition Rate. The objective of this project is to analyze Customer Attrition Rate of a Company using Machine-Learning. We visualize how each feature is related to our target variable and build a model which gives the chances of Attrition of each Customer in the Test dataset. As Customer retention becomes a valuable metric, it is in companies’ best interests to reduce Customer Attrition Rate. Companies that constantly monitor how people engage with products, encourage clients to share opinions, and solve their issues promptly have greater opportunities to maintain mutually beneficial client relationships. 

Similar model can also be used by HR's to predict Employee Attrition Rate (Turnover) in an Organization. It’s important to be aware of the implications of Attrition and how to avoid Employee Turnover.

## Requirements

[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)      [<img alt="Python" src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/>](https://python.org) 

[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg" width=170>](https://numpy.org)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/450px-Pandas_logo.svg.png" width=150>](https://pandas.pydata.org)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/7/73/Microsoft_Excel_2013-2019_logo.svg" height=75 width=150>](https://www.microsoft.com/en-in/microsoft-365/excel)

[<img target="_blank" src="https://matplotlib.org/_static/logo2_compressed.svg" width=170>](https://matplotlib.org)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[<img target="_blank" src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" width=150>](https://seaborn.pydata.org)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" width=150>](https://scikit-learn.org/stable)        


## Installation

This project requires Python 3.8. You will also need to have software installed to run and execute a Jupyter Notebook.
If you do not have Python installed yet, it is highly recommended that you install the Anaconda distribution of Python, which already has the required packages and more included. 

The project code is provided in the `Attrition_Rate_Analytics.ipynb` notebook file. You will also be required to use the included `Train_Data.csv` and `Test_Data.csv` dataset files. The final results of the project are saved in `Predictions.csv` file.

In a terminal or Anaconda Prompt window, navigate to the top-level project directory `Attrition_Rate_Analytics/` (that contains the project or notebook file) and run one of the following commands:

```bash
ipython notebook Attrition_Rate_Analytics.ipynb
```  
or
```bash
jupyter notebook Attrition_Rate_Analytics.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

## Conclusion

It is very critical for business to have an idea about why and when customers are likely to exit. Hence, through this analysis we build a model that Predicts the chances of Attrition of customers in a Telecom company which can help them reconsider, rebuild their products and change their business strategy accordingly to prevent customers from leaving the company.

## Future Work

We can always try to improve the model. The fuel of Machine Learning models is data so if we can collect more data, it is always helpful in improving the model. We can also try a wider range of parameters in GridSearchCV because a little adjustment in a parameter may slighlty increase the model.

Finally, we can try more robust or advanced models. Please keep in mind that there will be a trade-off when making such kind of decisions. Advanced models may increase the accuracy but they require more data and more computing power. So it comes down to business decision.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../master/LICENSE)
