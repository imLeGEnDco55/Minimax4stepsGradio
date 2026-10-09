# CONTEXT — MiniMax-H3 FlashGen 4-Step Colab Runner

> Departamento: +FlowCode Department · imLeGEnDco  
> Fecha: 2026-10-09  

---

## 🎯 ESTADO ACTUAL

Desarrollo y configuración de Notebook para Google Colab que clona y levanta la aplicación Gradio desde el repositorio personal [`imLeGEnDco55/Minimax4stepsGradio`](https://github.com/imLeGEnDco55/Minimax4stepsGradio) con exposición pública vía **Cloudflare Tunnel (`trycloudflare.com`)**.

---

## ⚙️ STACK & ARQUITECTURA

- **Repositorio Fuente**: `https://github.com/imLeGEnDco55/Minimax4stepsGradio`
- **Modelo Base**: `MiniMaxAI/MiniMax-H3` (Transformer omni-modal 33B para generación de video + audio estéreo sincronizado).
- **LoRA Distilled**: `Beidouqixing/minimax-h3-4step-lora-flashgen` (4 pasos de inferencia mediante DMD2 / VSD).
- **Split Pipeline**:
  - **Generator (Local Colab)**: `MiniMaxH3GeneratorBlocks` (Transformer 33B en bfloat16 + VAEs + Schedulers con sigmas fijados).
  - **Conditioner (Remoto HF Space)**: `multimodalart/qwen3vl-conditioner` (Qwen3-VL de 62 GB procesado vía Gradio API).
- **Túnel de Red**: Cloudflare Tunnel (`cloudflared`) apuntando a `http://127.0.0.1:7860`.
- **Requisitos de GPU en Colab**:
  - Obligatorio: **A100 (80GB)** en Colab Pro/Pro+ con High-RAM (~66 GB de VRAM ocupados por el Transformer).
  - Gated Model: Requiere `HF_TOKEN` con permiso de lectura tras aceptar la licencia en Hugging Face.

---

## 📁 ARCHIVOS PRINCIPALES

- [minimax_h3_flashgen_colab.ipynb](file:///e:/Appz/Minimax%20H3/minimax_h3_flashgen_colab.ipynb): Cuaderno interactivo configurado para clonar y ejecutar desde `imLeGEnDco55/Minimax4stepsGradio`.
- [generate_notebook.py](file:///e:/Appz/Minimax%20H3/generate_notebook.py): Generador programático del notebook JSON.
- [minimax-h3-flashgen-4step/](file:///e:/Appz/Minimax%20H3/minimax-h3-flashgen-4step): Copia local del Space original.

---

## ⚠️ ADVERTENCIAS ACTIVAS

1. **VRAM Limitation**: En entornos con menos de 80 GB de VRAM (T4, L4, V100), el modelo fallará con CUDA Out-Of-Memory (OOM).
2. **Hugging Face Gated Checkpoint**: El modelo base requiere autorización previa en Hugging Face antes de descargar los pesos.
