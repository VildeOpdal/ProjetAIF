#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import pandas as pd
from transformers import DistilBertModel, DistilBertTokenizerFast
from annoy import AnnoyIndex
from flask import Flask, jsonify, request, send_from_directory
import io

# Charger le modèle MobileNet V3 Small avec les poids par défaut
mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
model = torch.nn.Sequential(
    mobilenet.features, 
    torch.nn.AdaptiveAvgPool2d(output_size=1), 
    torch.nn.Flatten()
)
model.eval()

# Exporter le modèle sous le nom "model_3"
model_3 = model
