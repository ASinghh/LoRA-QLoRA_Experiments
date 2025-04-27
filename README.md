# LoRA Fine-tuning WizardLM-7B on SQuAD

This repository contains a Jupyter Notebook (`LoRA_finetune.ipynb`) demonstrating how to perform parameter-efficient fine-tuning (PEFT) using Low-Rank Adaptation (LoRA) on the `TheBloke/wizardLM-7B-HF` model. The fine-tuning task is Question Answering, using the SQuAD v1.1 dataset. The process utilizes 4-bit quantization (QLoRA) for memory efficiency during training.

## Features

*   **LoRA Fine-tuning:** Employs LoRA from the Hugging Face `peft` library to significantly reduce the number of trainable parameters.
*   **4-bit Quantization (QLoRA):** Uses `bitsandbytes` to load the base model in 4-bit precision, drastically reducing GPU VRAM requirements.
*   **SQuAD Dataset:** Downloads and preprocesses the Stanford Question Answering Dataset (SQuAD v1.1) into a suitable format for instruction fine-tuning.
*   **Hugging Face Ecosystem:** Leverages `transformers` for model and tokenizer loading, `datasets` for data handling, and `peft` for LoRA implementation.
*   **Model Merging:** Includes steps to merge the trained LoRA adapters back into the base model for standalone deployment.

## Prerequisites

*   **Python:** Python 3.8+ recommended.
*   **GPU:** An NVIDIA GPU with sufficient VRAM is required (e.g., >= 16GB, but more is better). CUDA toolkit compatible with the installed PyTorch and `bitsandbytes` versions.
*   **Libraries:** Python dependencies listed in `requirements.txt`.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure your PyTorch installation matches your CUDA version. You might need to install PyTorch separately following instructions on the [official PyTorch website](https://pytorch.org/).*

## Dataset

The script automatically downloads the SQuAD v1.1 training dataset (`train-v1.1.json`) from its official source URL. It then preprocesses this JSON data into a JSON Lines (`.jsonl`) file (`simple_squad.jsonl`) containing `{"question": "...", "answer": "..."}` pairs. This `.jsonl` file is then loaded using the `datasets` library.

## Usage

1.  **Launch Jupyter:** Start a Jupyter Notebook or JupyterLab server in your environment.
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```

2.  **Open and Run:** Open the `LoRA_finetune.ipynb` notebook.

3.  **Configure (Optional):** Review the parameters in the configuration cell (cell ID `740b1bcd-...`). You might want to adjust:
    *   `max_length`
    *   LoRA parameters (`lora_alpha`, `lora_dropout`, `lora_r`)
    *   Training arguments (`output_dir`, `learning_rate`, batch sizes, `save_steps`, etc.)

4.  **Execute Cells:** Run the notebook cells sequentially.
    *   The script will download the dataset, preprocess it, load the tokenizer and quantized base model, apply LoRA, and start the training process using the Hugging Face `Trainer`.
    *   Model weights and tokenizer files will be cached locally in the `./models` directory.

5.  **Training:** Monitor the training progress. Checkpoints will be saved periodically to the specified `output_dir` (default: `outputs_squad`).

6.  **Merging (Optional):** After training, the final cells demonstrate how to:
    *   Load the base model again.
    *   Load the trained LoRA adapter weights from a checkpoint (update `lora_path` if needed).
    *   Merge the adapter weights into the base model.
    *   Save the fully merged model to a specified path (default: `outputs_squad/merged_model`).

## Output

*   **LoRA Adapters:** Saved during training in `outputs_squad/checkpoint-<step>/`. These contain only the adapter weights, not the full model.
*   **Merged Model:** If the merging cells are run, the full fine-tuned model (base model + merged adapters) is saved in `outputs_squad/merged_model/`. This can be loaded directly using `AutoModelForCausalLM.from_pretrained("outputs_squad/merged_model")`.

## Notes

*   **GPU Memory:** 4-bit quantization significantly reduces memory usage, but a 7B parameter model still requires substantial VRAM. Monitor your GPU memory usage.
*   **Training Time:** Fine-tuning time depends heavily on the GPU, dataset size, and training configuration (batch size, number of epochs/steps).
*   **Model Caching:** Hugging Face models and datasets are cached (default: `./models` in this script) to speed up subsequent runs.
*   **BitsAndBytes/CUDA:** Ensure `bitsandbytes` is compatible with your CUDA toolkit version. Mismatches can cause errors during model loading or training.

## Acknowledgements

The code in the `LoRA_finetune.ipynb` notebook is heavily based on and adapted from the excellent work found in the [gmongaras/Wizard_QLoRA_Finetuning](https://github.com/gmongaras/Wizard_QLoRA_Finetuning) repository. Credit goes to the original author for providing a clear example of QLoRA fine-tuning.

## License

MIT License
