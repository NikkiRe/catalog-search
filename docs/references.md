# Primary sources

The project implements a testable attribute representation on top of classical
retrieval and learning to rank. It does not claim to introduce a new learning
algorithm or reproduce a published leaderboard.

- [Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search](https://arxiv.org/abs/2206.06588),
  Reddy et al., 2022. Defines E/S/C/I and the graded ranking task.
- [ESCI source and Apache-2.0 license](https://github.com/amazon-science/esci-data/tree/7916cdf6ab75a462e77f20ab40428a10923998d5).
  Source byte hashes and actual released counts are recorded in the data manifest.
- [WANDS: Dataset for Product Search Relevance Assessment](https://easychair.org/publications/preprint/j2D4),
  Chen et al., 2022. Furniture/home-goods transfer with a different label vocabulary.
- [WANDS source and MIT license](https://github.com/wayfair/WANDS/tree/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5).
  The adapter's main set excludes 14 contradictory duplicate query-product pairs.
- [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.nowpublishers.com/article/Details/INR-019),
  Robertson and Zaragoza, 2009. BM25 term saturation and document-length normalization.
- [From RankNet to LambdaRank to LambdaMART: An Overview](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/),
  Burges, 2010. Pairwise gradient-based learning to rank.
- [LightGBM ranking parameters](https://lightgbm.readthedocs.io/en/v4.7.0/Parameters.html#label_gain).
  Explicit label gains are necessary: default ordinal/exponential gains are not
  the ESCI utility map used here.

The dataset papers support task definitions and provenance. Model results in
this repository come from the saved experiment, not the papers.
