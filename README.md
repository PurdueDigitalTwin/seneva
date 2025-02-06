<!-- markdownlint-disable -->

# Quantifying Uncertainty in Motion Prediction with Variational Bayesian Mixture

<div align="center">
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/PurdueDigitalTwin/seneva"><img alt="Template" src="https://img.shields.io/badge/-SeNeVA-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
</div>

This repository contains the official implementation of the CVPR 2024 paper ["Quantifying Uncertainty in Motion Prediction with Variational Bayesian Mixture"](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_Quantifying_Uncertainty_in_Motion_Prediction_with_Variational_Bayesian_Mixture_CVPR_2024_paper.html).

Authors: Juanwu Lu, Can Cui, Yunsheng Ma, Aniket Bera, Ziran Wang

## Abstract

Safety and robustness are crucial factors in developing trustworthy autonomous vehicles. One essential aspect of addressing these factors is to equip vehicles with the capability to predict future trajectories for all moving objects in the surroundings and quantify prediction uncertainties. In this paper we propose the Sequential Neural Variational Agent (SeNeVA) a generative model that describes the distribution of future trajectories for a single moving object. Our approach can distinguish Out-of-Distribution data while quantifying uncertainty and achieving competitive performance compared to state-of-the-art methods on the Argoverse 2 and INTERACTION datasets. Specifically a 0.446 meters minimum Final Displacement Error a 0.203 meters minimum Average Displacement Error and a 5.35% Miss Rate are achieved on the INTERACTION test set. Extensive qualitative and quantitative analysis is also provided to evaluate the proposed model.

## Citation

If you find our work helpful for your project, please cite

```bibtex
@InProceedings{Lu_2024_CVPR,
    author    = {Lu, Juanwu and Cui, Can and Ma, Yunsheng and Bera, Aniket and Wang, Ziran},
    title     = {Quantifying Uncertainty in Motion Prediction with Variational Bayesian Mixture},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {15428-15437}
}
```

## License

This implementation of SeNeVA is licensed under the MIT license. See `LICENSE` for details.

