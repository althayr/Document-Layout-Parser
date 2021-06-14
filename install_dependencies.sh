
# OCR + PDF Reader system level dependencies
sudo apt-get install -y poppler-utils tesseract-ocr

# Python lib dependencies
pip install sentencepiece pyyaml==5.1
pip install -f https://download.pytorch.org/whl/torch_stable.html \
            datasets==1.6.2 \
            torch==1.8.0+cu111 \
            torchvision==0.9.0+cu111 \
            transformers==4.5.1 \
            seqeval==1.2.2 \
            torchsummary==1.5.1 \
            torchtext==0.9.0
pip install fastai==1.0.61 \
             'git+https://github.com/facebookresearch/detectron2.git'

pip install -U argparse pytesseract opencv-python gradio scikit-image numpy pdf2image

# Installing LayoutLM family libraries 
rm -rf unilm
git clone https://github.com/microsoft/unilm
cd /unilm/layoutlmft
pip install .
