# Book Rating Prediction
As a part of the course Text Mining [TDDE16], I'm attempting to predict the rating a user has given a book given its review, using `distilBERT-base-cased`. 

To run the code, first clone the project onto your own PC. Then relocate to the root of the project and run `pip install -e .` to install the `PredictRating` package. Once the package has been installed, `pytorch` has to be installed separately in order to fit your system. See [this](https://pytorch.org/get-started/locally/) for an install command that fits your specification.

All scripts run from root. Download the goodreads datasets from [here](https://mengtingwan.github.io/data/goodreads) (the one called `goodreads_reviews_spoiler_raw.json.gz`), place it in `book-rating-prediction/data` and name it `reviews.json`. For the Amazon set, download it from [here](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews), place it in `book-rating-prediction/data` and name it `reviews_amzn.csv`.

A model is trained with `train.py` and evaluated with `bertstats.py`. `baseline.py` and `humanlevel.py` creates baselines for the BERT classifier. `constants.py` contains all hyperparameter values used during training. There are some tests written for the package in `book-rating-prediction/tests`.

The BERT model from the report can be downloaded from [here](bit.ly/41EqddH). Download it and place it in `book-rating-prediction/models` if you want to use it.

Final report awarded with highest grade (5) available @ [here](https://bit.ly/3QCNGph).
