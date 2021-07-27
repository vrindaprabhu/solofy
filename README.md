# SOLOFY

## Purpose 
The is an initial version of the repo which can be used to reduce the presence of people from the photos. 
In other words, make the picture fully focussed on the selected person - hence the name - _Solofy_.

Has only been tested on CPU.

## Running
Run the below command for an interactive execution -

- `python main.py` - Runs the code on the _inimage.png_ file saved in the __input__ folder. Output is saved as _final.png_ in the __output__ folder.
- `streamlit run demo.py` - Runs a minimalistic streamlit demo for solofy.

__NOTE:__
- The models have to be downloaded into the models directory. The links are present in tha README in the folder.

## Things to do
- [x] Make an end-to-end pipeline.
- [x] Move configurations to a single location. All configurations now in __solofy.py__ under `config` folder.
- [x] Have a streamlit demo.
- [x] Add requirements.txt.
- [ ] Handle exceptions more effectively.
- [ ] Test for GPU support and higher resolution image.
- [x] Use multiprocessing.
- [ ] Use the [rembg](https://pypi.org/project/rembg/) package and check for speed _(uses u2-net in the backend)_.
- [ ] Make the selection of salient background _finer_. Can also think of using matting.

## Acknowledgement
This repo is just multiple awesome repositories put in together. The link to those repositories are as below:
- [Yolov3](https://github.com/zhaoyanglijoey/yolov3)
- [U2Net](https://github.com/xuebinqin/U-2-Net)
- [DeepFillv2](https://github.com/csqiangwen/DeepFillv2_Pytorch)


## Other Info
- Check [image denoising](https://github.com/topics/image-denoising) and decrappification _(fastai)_ to see if inpainted image can be improved.
- [Fine grained image inpainting](https://github.com/researchmm/AOT-GAN-for-Inpainting) seems to be SOTA for inpainting. Unfortunately unable to complete a run on CPU.
- Performs REALLY bad when the person is very prominent and close in the foreground. Should deep dive and check why.
