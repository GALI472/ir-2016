# ir-2016
CS 572 - Information Retrieval final project, Emory University 2016

## Resources
- [Semantic Text Similarity Dataset Hub](https://github.com/brmson/dataset-sts)
- [Trec LiveQA 2016](https://sites.google.com/site/trecliveqa2016/)
- [Java Skeleton Server](https://github.com/yuvalpinter/LiveQAServerDemo)

## Data
- L6 from [Yahoo Webscope](http://webscope.sandbox.yahoo.com/)
- [Wikipedia Dump](https://dumps.wikimedia.org/)
- [Reddit Comment Dump](https://bigquery.cloud.google.com/welcome/eco-serenity-126001?pli=1) (need to figure out how to link comments with posts)

## Todo
- [x] Serialize documents in Yahoo QA XML files as an SQLite database
- [ ] Build HTTP server for recieving questions for the competition: See Java example on webpage
- [x] Make interface for adding generic prediction models
- [ ] Write mixture of experts predictor given all prediction models models
- [ ] Article summarizer (use on Wikipedia, Reddit to add more recent data)

#### Prediction models

###### Unsupervised

- [x] Latent Dirichlet Allocation - [Gensim](https://radimrehurek.com/gensim/models/ldamodel.html)
- [x] Latent Semantic Indexing - [Gensim](https://radimrehurek.com/gensim/models/lsimodel.html)
- [x] TF-IDF or BM-25 - [Gensim](https://radimrehurek.com/gensim/models/tfidfmodel.html)
- [x] Word2Vec - [Gensim](https://radimrehurek.com/gensim/models/word2vec.html)

###### Learn to Rank

- [ ] Neural network model - Some success using Keras
