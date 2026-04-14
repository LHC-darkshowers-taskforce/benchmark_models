# Benchmark models

## Running CMS benchmark cards

### Hadronic semi-visible jets
Model used in [CMS-EXO-19-020](https://arxiv.org/abs/2112.11125), based on pheno paper [arXiv:1503.00009](https://arxiv.org/abs/1503.00009).

The code ```svjHelper_hadronic.py``` takes the following arguments:
  * ```--mZprime```: Z' mass (GeV) (required, search: >= 1.5 TeV)
  * ```--mDark```: dark hadron mass (GeV) (required, suggested: 20 GeV)
  * ```--rinv```: effective invisible fraction (required, suggested: 0.3, 0.5)
  * ```--alpha```: dark sector coupling strength (required, suggested: peak). Values: peak, low, high. ```Peak``` is the value of alpha that maximises the multiplicities, ```high = 1.5*peak```, ```low = 0.5*peak```.
  * ```--lambda```: dark confinement scale (GeV) (required in alternative to alpha).

In order to produce your own card use commands like:

```
python svjHelper.py --mZprime 3000.0 --rinv 0.3 --mDark 20 --alpha peak
```

### Emerging jets
Model used in [CMS-EXO-22-015](https://arxiv.org/abs/2403.01556), based on pheno paper [arXiv:1803.08080](https://arxiv.org/abs/1803.08080).

The code ```emjHelper.py``` takes the following arguments:
  * ```--mMed```: Mediator mass (GeV) (default: 1000 GeV, , search: >= 1 TeV)
  * ```--kappa```: dark quark - SM quark yukawa coupling squared factor (scaling lifetimes) (default: 1)
  * ```--mDark```: dark hadron mass (GeV) (default: 10 GeV)
  * ```--type```: alignment to SM up/down quarks (default: down)
  * ```--mode```: mixing scenarios to simulate (default: aligned)
  * ```--channel```: t or s channel for production (default: t)


In order to produce your own card use commands like:

```
python emjHelper.py --mMed 1000.0 --mDark 10 --kappa 0.36 --type down --mode aligned --channel t
```

### Flavour-democratic semi-visible jets
Model used in [CMS-EXO-24-029](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/EXO-24-029/index.html), based on pheno paper [arXiv:2206.03909](https://arxiv.org/abs/2206.03909).

The code ```svjHelper_leptons_democratic.py``` takes the following arguments:
  * ```--mZprime```: Z' mass (GeV) (required, search: >= 1.5 TeV) 
  * ```--rinv```: effective invisible fraction (required, suggested: 0.3, 0.5)
  * ```--mPiOverLambda```: ratio of the lightest pseudo-scalar meson and dark confinement scale (required, suggested: 1.6)
  * ```--lambda```: dark confinement scale (GeV) (required, suggest: 10 GeV).

In order to produce your own card use commands like:

```
python svjHelper_leptons_democratic.py --mZprime 3000.0 --rinv 0.3 --mPiOverLambda 1.6 --lambda 10
```

### Tau-enriched semi-visible jets
Model used in [CMS-EXO-26-005](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/EXO-26-005/), based on pheno paper [arXiv:2212.11523](https://arxiv.org/abs/2212.11523).

The code ```svjHelper_taus.py``` takes the following arguments:
  * ```--mZprime```: Z' mass (GeV) (required, search: >= 1.5 TeV)
  * ```--rinv```: effective invisible fraction (required, suggested: 0.3, 0.5)
  * ```--brtau```: effective branching fraction of dark pions to tau leptons (required, suggested: 0.3, 0.5, 0.7)
  * ```--mPiOverLambda```: ratio of the lightest pseudo-scalar meson and dark confinement scale (required, suggested: 0.8)
  * ```--lambda```: dark confinement scale (GeV) (suggest: 10 GeV).

In order to produce your own card use commands like:

```
python svjHelper_taus.py --mZprime 3000.0 --rinv 0.3 --brtau 0.5 --mPiOverLambda 0.8 --lambda 10 
```


