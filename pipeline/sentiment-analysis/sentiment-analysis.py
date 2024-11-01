from transformers import pipeline

base_model = "finiteautomata/bertweet-base-sentiment-analysis"
fine_tuned_model = "fine_tuned_models/sentiment-analysis/model"
# c_path = "fine_tuned_models/sentiment-analysis/base_model"
sentiment_pipeline = pipeline(model=base_model)
# sentiment_pipeline = pipeline(model=fine_tuned_model,task="sentiment-analysis")
data = ["I love you", "I hate you","You don't know you can do this"]
data = sentiment_pipeline(data)
print(data)