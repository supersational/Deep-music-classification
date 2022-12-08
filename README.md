# Deep-music-classification

This repository contains a pytorch implentation of several neural network architectures used to classify music genre.


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
