# ir-2016
CS 572 - Information Retrieval final project, Emory University 2016

## Data
- L6 from [Yahoo Webscope](http://webscope.sandbox.yahoo.com/)

## Todo
- [x] Serialize documents in Yahoo QA XML files as an SQLite database
- [ ] Write interface for recieving questions for the competition: See Java example on webpage
- [ ] Make interface for adding generic prediction models
- [ ] Write mixture of experts predictor given all prediction models models
- [ ] Implement prediction models, see if they work

#### Prediction models
- [ ] Latent Dirichlet Allocation - [Gensim](https://radimrehurek.com/gensim/models/ldamodel.html)
- [ ] Latent Semantic Indexing - [Gensim](https://radimrehurek.com/gensim/models/lsimodel.html)
- [ ] TF-IDF or BM-25 - [Gensim](https://radimrehurek.com/gensim/models/tfidfmodel.html)
- [ ] Word2Vec - [Gensim](https://radimrehurek.com/gensim/models/word2vec.html)
- [ ] Neural network model - Some success using Keras
