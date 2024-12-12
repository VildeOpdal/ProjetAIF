#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
#import torchvision.models as models
import pandas as pd
from transformers import DistilBertModel, DistilBertTokenizerFast
import numpy as np
import gradio as gr
import torch
import requests
from annoy import AnnoyIndex
from flask import Flask, jsonify, request, send_from_directory
import io


# Charger le modèle DistilBERT
model_2 = DistilBertModel.from_pretrained('distilbert-base-uncased')
