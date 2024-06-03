# Recommender System

Exploration of data and cleaning is done in the notebook "Data Exploration and Cleaning.ipynb"

Tooling around to explore the architectures is done in the notebook "Beginning.ipynb".

Finally everything is compiled in a couple of scripts.

# Data

Please, download the RetailRocket ecommerce dataset manually from here: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset/data, since it's too big to store on GitHub.

# How to Run

* To train 'collaborative_recommender.py' do:
`python ./collaborative_recommender.py`

    The script goes in schedule mode where the model is retrained at midnight, but you can Ctrl+C to exit the script.

* To train 'content_based_cosine_sim.py' do:
`python ./content_based_cosine_sim.py`

Both are automatically saved in cosine_sim.pkl and recommender_model.pth.

* To deploy do:
`python ./deploy.py`

Then you can do requests like:

`curl -X POST -H "Content-Type: application/json" -d '{"user_id": 1}' http://127.0.0.1:5000/recommend`

Example output:

        {
            "recommendations": [
                107153,
                136606,
                262406,
                454338,
                49611,
                82817,
                381507,
                16208,
                350499,
                351456
            ],
            "response_time": 0.005533933639526367
        }