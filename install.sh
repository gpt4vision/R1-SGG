


pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers@git+https://github.com/huggingface/transformers.git@2c2495cc7b0e3e2942a9310f61548f40a2bc8425
pip install trl@git+https://github.com/huggingface/trl.git@dae3df16cadb301721e85ad8e6e7c2c49685c23c

pip install --upgrade --no-build-isolation flash-attn==2.7.4.post1

pip install -r requirements.txt

pip install -e .
