# Adversarial Robustness of Contemporary Machine Learning Models.

This is my final project repository for the CMPUT 664 course.
In this work, we have implemented three different adversarial attacks on four different datasets and the model hardening defense mechanism. We leveraged the open-source tool, `Adversarial Robustness Toolbox`, to generate these attacks. Check their [github repository](https://github.com/Trusted-AI/adversarial-robustness-toolbox) to know more about the installation details.

This repository includes:

* a Dockerfile to build the Docker script,
* the instructions to reproduce the results
* `python` source codes to generate:
  * FGSM and PGD attacks on text dataset
  * Defense mechanism: model hardening with PGD on text dataset
  * DeepFool attack on audio dataset
  * Defense mechanism: model hardening with PGD on audio dataset
  * FGSM and DeepFool attacks on image dataset (Fruits-360, Model: CNN, PyTorch)
  * FGSM and PGD on image dataset (Fruits-360, Model: sequential NN, tf.keras)
  * FGSM and DeepFool on image dataset (MNIST, Model: sequential NN, tf.keras)
  * Output files, graphs, plots, and output images

## Docker Image
A pre-built version of this project is available as Docker image. To reproduce the experimental results using the image:
```
git pull <this repository>
docker build -t <image_name>
docker run -i <image_name>
```

A demo video showing how to reproduce the experiments is available in the 
## Adversarial Threats

<p align="center">
  <img src="docs/images/adversarial_threats_attacker.png?raw=true" width="400" title="ART logo">
  <img src="docs/images/adversarial_threats_art.png?raw=true" width="400" title="ART logo">
</p>
<br />

## ART for Red and Blue Teams (selection)

<p align="center">
  <img src="docs/images/white_hat_blue_red.png?raw=true" width="800" title="ART Red and Blue Teams">
</p>
<br />

## Learn more

| **[Get Started][get-started]**     | **[Documentation][documentation]**     | **[Contributing][contributing]**           |
|-------------------------------------|-------------------------------|-----------------------------------|
| - [Installation][installation]<br>- [Examples](examples/README.md)<br>- [Notebooks](notebooks/README.md) | - [Attacks][attacks]<br>- [Defences][defences]<br>- [Estimators][estimators]<br>- [Metrics][metrics]<br>- [Technical Documentation](https://adversarial-robustness-toolbox.readthedocs.io) | - [Slack](https://ibm-art.slack.com), [Invitation](https://join.slack.com/t/ibm-art/shared_invite/enQtMzkyOTkyODE4NzM4LTA4NGQ1OTMxMzFmY2Q1MzE1NWI2MmEzN2FjNGNjOGVlODVkZDE0MjA1NTA4OGVkMjVkNmQ4MTY1NmMyOGM5YTg)<br>- [Contributing](CONTRIBUTING.md)<br>- [Roadmap][roadmap]<br>- [Citing][citing] |

[get-started]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/Get-Started
[attacks]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Attacks
[defences]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Defences
[estimators]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Estimators
[metrics]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Metrics
[contributing]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/Contributing
[documentation]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/Documentation
[installation]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/Get-Started#setup
[roadmap]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/Roadmap
[citing]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/Contributing#citing-art

The library is under continuous development. Feedback, bug reports and contributions are very welcome!

# Acknowledgment
This material is partially based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under
Contract No. HR001120C0013. Any opinions, findings and conclusions or recommendations expressed in this material are
those of the author(s) and do not necessarily reflect the views of the Defense Advanced Research Projects Agency (DARPA).
