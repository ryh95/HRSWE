# HRSWE

This is the repository that contains the code of "Specializing Word Vectors by Spectral Decomposition on Heterogeneously Twisted Graphs".


## How to reproduce the results

Word Similarity and Synonym/Antonym Classification task

0. Download the [data](https://www.dropbox.com/s/q66b1j8f0fuodsx/data.zip?dl=0)
1. Download [sim_results](https://www.dropbox.com/s/chqg0psxnu0tnl1/sim_results.zip?dl=0) and [clf_results](https://www.dropbox.com/s/mrfvjnfs91x8561/clf_results.zip?dl=0)
2. Import `results.pickle` with python `pickle` module
3. Change hrswe/ar config in `config.py` with `config` in `results.pickle`
4. Run `main.py` or `adv_main.py`

Lexical Simplification task

0. Download the [data](https://www.dropbox.com/s/q66b1j8f0fuodsx/data.zip?dl=0)
1. Open `results` in `lexical_simiplification` directory and load `res-hyp.pickle` with `skopt` module
2. Change hrswe/ar config in `main_sp_lex_mturk.py` with `config` in `res-hyp.pickle`
3. Run `main_sp_lex_mturk.py`

Note: Due to the randomness in the hyperparameter tuning method, you may not produce the exact same results. But you should obtain similar results.

## Supplementary Materials

TODO:

- [ ] add the paper
- [ ] add the poster 
- [ ] add the bib file 
