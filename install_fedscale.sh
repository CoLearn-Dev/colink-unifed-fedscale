 #!/bin/bash
 #!/usr/bin/env python
 git clone https://github.com/xmhuangzhen/FedScale.git
 cd FedScale
 git checkout 0328update
 source ./install.sh
 pip install -e .
 cd ..

##### conda create -n colink-protocol-unifed-fedscale python=3.9.16 -y
##### conda activate colink-protocol-unifed-fedscale
 pip install --upgrade pip
 pip install flbenchmark
 pip install nest_asyncio
 pip install pyyaml
 pip install colink
 pip install pytest
 pip install -e .
 
 