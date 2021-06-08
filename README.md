# Large-Scale Unsupervised Object Discovery

Huy V. Vo, Elena Sizikova, Cordelia Schmid, Patrick Pérez, Jean Ponce

[[paper]]()

### Abstract
Existing approaches to unsupervised object discovery (UOD) do not scale up to large datasets without approximations which compromise their performance. We propose a novel formulation of UOD as a ranking problem, amenable to the arsenal of distributed methods available for eigenvalue problems and link analysis. Extensive experiments with [COCO](https://cocodataset.org/#home) and [OpenImages](https://storage.googleapis.com/openimages/web/index.html) demonstrate that, in the single-object discovery setting where a single prominent object is sought in each image, the proposed LOD (Large-scale Object Discovery) approach is on par with, or better than the state of the art for medium-scale datasets (up to 120K images), and over 37% better than the only other algorithms capable of scaling up to 1.7M images. In the multi-object discovery setting where multiple objects are sought in each image, the proposed LOD is over 14% better in average precision (AP) than all other methods for datasets ranging from 20K to 1.7M images.

![Teaser](images/teaser.jpg)

Code will be released soon!

## Quantitative results
![Quantitative result][images/quantitative_results.jpg]

## Citations

```
@misc{Vo21LOD,
  title     = {Large-Scale Unsupervised Object Discovery},
  author    = {Vo, Huy V. and Sizikova, Elena and Schmid, Cordelia and P{\'e}rez, Patrick and Ponce, Jean},
  year      = {2021},
  eprint={},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## Acknowledgments

This work was supported in part by the Inria/NYU collaboration, the Louis Vuitton/ENS chair on artificial intelligence and the French government under management of Agence Nationale de la Recherche as part of the “Investissements d’avenir” program, reference ANR19-P3IA-0001 (PRAIRIE 3IA Institute). Huy V. Vo was supported in part by a Valeo/Prairie CIFRE PhD Fellowship.