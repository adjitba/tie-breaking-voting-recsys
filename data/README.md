# Datasets

The datasets are publicly available but are not redistributed in this repository, because their licenses do not allow it or ask users to obtain them from the original source. Download them and place the files as follows.

| Dataset | Source | Expected file |
| --- | --- | --- |
| MovieLens 100K | GroupLens, <https://grouplens.org/datasets/movielens/100k/> (F.M. Harper, J.A. Konstan, ACM TiiS 5(4), 2015) | `data/ml-100k/u.data` |
| FilmTrust | LibRec datasets, <https://guoguibing.github.io/librec/datasets.html> (G. Guo, J. Zhang, N. Yorke-Smith, IJCAI 2013) | `data/filmtrust/ratings.txt` |
| Amazon Digital Music (5-core) | Amazon Review Data 2018, <https://nijianmo.github.io/amazon/index.html> (J. Ni, J. Li, J. McAuley, EMNLP-IJCNLP 2019) | `data/amazon-digital-music/Digital_Music_5.json` |

Notes:

- MovieLens 100K: extract `ml-100k.zip` and copy `u.data` (tab-separated `user item rating timestamp`).
- FilmTrust: `ratings.txt` with space-separated `user item rating`. Duplicate user–item pairs are reduced to the last one listed.
- Amazon: download the 5-core file `Digital_Music_5.json.gz` and decompress it to `Digital_Music_5.json` (one JSON object per line). Duplicate reviews are reduced to the most recent one; user and item IDs are mapped to integers.

Filtering (Table 3 of the paper) is applied by the code: users and items below the minimum number of ratings (ML-100K 20/5, FilmTrust 5/5, Amazon 5/5) are removed repeatedly.
