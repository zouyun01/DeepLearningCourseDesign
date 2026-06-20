#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render display equations for the method-principles section (2.1) as clean
PNGs (python-docx cannot author OMML):
  grpo_formula.png  group-relative advantage + clipped objective + ratio
  sft_formula.png   supervised autoregressive (teacher-forcing) loss
Run: E:/anaconda3/envs/dl/python.exe scripts/plot_formula.py
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"mathtext.fontset": "stix", "font.family": "serif",
                     "savefig.bbox": "tight", "figure.facecolor": "white"})
OUT = Path("results/figures")


def grpo():
    fig = plt.figure(figsize=(7.8, 2.25))
    fig.text(0.5, 0.84,
             r"$A_i=\dfrac{r_i-\mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)+\epsilon}$",
             ha="center", va="center", fontsize=18)
    fig.text(0.5, 0.50,
             r"$\mathcal{J}(\theta)=\mathbb{E}\!\left[\dfrac{1}{G}\sum_{i}"
             r"\min\!\left(\rho_i A_i,\ \mathrm{clip}(\rho_i,1-\varepsilon,1+\varepsilon)\,A_i\right)\right]"
             r"-\beta\,\mathrm{KL}\!\left(\pi_\theta\,\Vert\,\pi_{\mathrm{ref}}\right)$",
             ha="center", va="center", fontsize=18)
    fig.text(0.5, 0.16,
             r"$\rho_i=\pi_\theta(y_i\mid x)\,/\,\pi_{\theta_{\mathrm{old}}}(y_i\mid x)$",
             ha="center", va="center", fontsize=18)
    fig.savefig(OUT / "grpo_formula.svg")
    fig.savefig(OUT / "grpo_formula.png", dpi=300)
    plt.close(fig)


def sft():
    fig = plt.figure(figsize=(6.6, 0.9))
    fig.text(0.5, 0.5,
             r"$\mathcal{L}_{\mathrm{SFT}}(\theta)=-\sum_{t=1}^{|y|}"
             r"\log \pi_\theta\!\left(y_t \mid x,\ y_{<t}\right)$",
             ha="center", va="center", fontsize=18)
    fig.savefig(OUT / "sft_formula.svg")
    fig.savefig(OUT / "sft_formula.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    grpo(); sft()
    print("Saved results/figures/grpo_formula.png, sft_formula.png")
