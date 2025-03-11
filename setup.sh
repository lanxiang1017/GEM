conda create -n gem python=3.10 -y
conda activate gem
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install ftfy
pip install wfdb
pip install peft==0.10.0