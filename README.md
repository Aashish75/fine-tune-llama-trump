# fine-tune-llama-trump

Fine Tuning of Llama-3.2-1B-Instruct to improve it's ability to tweet like Donald Trump. 

Tweet Data from https://drive.google.com/file/d/1xRKHaP-QwACMydlDnyFPEaFdtskJuBa6/view?usp=sharing to fine tune the model. 

The model was fine-tuned with instruction-tweet pairs where the instruction asked the model to talk like Donald Trump and the tweet was a sample from the data mentioned above. 

LoRA(Low-Rank Adaptation) Config was used where tunable rank 16 matrices were induced instead of changing fine-tuning the Llama architecture itself.

Scaled training across multiple GPU instances using DeepSpeed on Amazon SageMaker; merged and pushed model to Hugging Face Hub.

**Model available at** - https://huggingface.co/Aashish75/llama-3.2-1b-trump0

