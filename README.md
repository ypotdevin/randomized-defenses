This repository accompanies the publication [An Empirical Investigation of Randomized Defenses against Adversarial Attacks](https://arxiv.org/abs/1909.05580) by Yannik Potdevin, Dirk Nowotka and Vijay Ganesh. It contains implementations of the presented defense mechanisms to protect deep neural networks against [adversarial examples](https://arxiv.org/abs/1312.6199).
In more detail: It contains implementations of
 * the L1 and L* defense mechanisms proposed by Gu and Rigazio in [Towards deep neural network architectures robust to adversarial examples](http://arxiv.org/abs/1412.5068)
 * our proposed adaptation of L1 and L* which we call L+
 * our proposed defense mechanism which we call RPENN

The reason we publish our source code is to enhance reproducibility of our results and to clarify implementation details which might be not discussed extensively within the publication.

---

Our code makes directly or inderectly use of the following third-party libraries (non-exhaustive list):
 * [Keras](https://keras.io/) (MIT license)
 * [scikit-learn](https://scikit-learn.org/) (New BSD license)
 * [NumPy](https://numpy.org/) (BSD license)
 * [Pillow](https://python-pillow.org/) (PIL Software License)
