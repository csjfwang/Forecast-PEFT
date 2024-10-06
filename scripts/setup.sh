pip3 --no-cache-dir install \
     torch==1.11.0+cu113 \
     torchvision==0.12.0+cu113 \
     --extra-index-url https://download.pytorch.org/whl/cu113

pip3 install natten==0.15.1+torch220cu121 -f https://shi-labs.com/natten/wheels/

pip install -r ./requirements.txt
pip install av2