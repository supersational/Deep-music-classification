# Deep-music-classification

This repository contains a pytorch implentation of several neural network architectures used to classify music genre.

# Data

Our code uses the GTZAN dataset [2], this can be placed in the `data` folder manually, or if not present the `dataset.py` has code that will automatically download it.  

# Usage

To replicate the configuration used by Schindler et al. [1] for the 'shallow' network run `main.py` with no arguments:
```bash
cd src
python main.py
```

To specify which model use the `--model` parameter, with one of `shallow` `deep` or `filter`.

```bash
python main.py --model deep
```

Our experiments found that removing the L1 normalisation, dropout, and LeakyReLU could improve performance. To replicate those results run:
```bash
python -u main.py --dropout 0.0 --alpha 0.0 --l1 0.0  
```

# Results

Our code performs validation every 10 epochs, and plots output every 100 epochs. It will also save all computed losses and accuracies (for training and validation) used to make the plots after it has finished running.

# References 
[1] A. Schindler, T. Lidy, and A. Rauber, “Comparing shal-
low versus deep neural network architectures for auto-
matic music genre classification,” 11 2016.


[2] T.  Ozseven and B. E.  Ozseven, “A content analysis of
the research approaches in music genre recognition,” in
2022 International Congress on Human-Computer Interac-
tion, Optimization and Robotic Applications (HORA), 2022,
pp. 1–13
