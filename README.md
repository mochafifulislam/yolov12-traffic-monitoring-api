# 🚗 Intelligent Traffic Monitoring System (YOLOv12 + DeepSORT)

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://docker.com)

Sistem Pemantauan dan Perhitungan Kendaraan Otomatis berbasis **YOLOv12**, **DeepSORT**, dan **FastAPI Backend** yang dikembangkan dari dataset CCTV riil persimpangan Kota Malang.

## 🎯 Key Achievements
- **Model Accuracy:** Overall mAP50 = **97.7%**, mAP50-95 = **74.9%**, Precision = **97.1%**.
- **Inference Speed:** **4.8 ms / frame** pada GPU Tesla T4.
- **Publication:** Proyek dikembangkan sebagai bagian dari tugas akhir S1 Matematika Universitas Brawijaya.

## 🏗️ Architecture & Features
1. **Roboflow MLOps Pipeline:** Versioning dataset otomatis (Version 4) terintegrasi via API.
2. **Multi-Object Tracking & Counting:** Integrasi YOLOv12 + DeepSORT dengan fitur Virtual Line Crossing.
3. **Production REST API:** Backend FastAPI pendukung Batch Video Processing & Live MJPEG Streaming.

## 📌 API Endpoints
- `GET /health` : Status ketersediaan model & akselerasi GPU.
- `POST /count-vehicles` : Memproses video CCTV & mengembalikan video ter-anotasi + metrik statistik JSON.
- `GET /stream/{job_id}` : Live MJPEG Video Streaming untuk konsumsi antarmuka frontend.

## 🚀 Quickstart (Docker Container)
```bash
docker build -t traffic-ai-api .
docker run -p 8000:8000 traffic-ai-api
