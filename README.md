## [NeurIPS 2021] Shape from Blur: Recovering Textured 3D Shape and Motion of Fast Moving Objects
### [YouTube](https://youtu.be/hPYWh9KGiu8) | [arXiv](https://arxiv.org/abs/2106.08762)

<img src="examples/imgs/aerobie.gif" height="160"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="examples/imgs/football.gif" height="160">  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="examples/imgs/vol.gif" height="160"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  <img src="examples/imgs/key__new.gif" height="160">

### Prerequisites 
Kaolin is available here: https://github.com/NVIDIAGameWorks/kaolin


### Running
![Examples](examples/imgs/sfb.png)

The code can be easily run by:
```bash
python optimize.py
```

Running with your inputs:
```bash
python optimize.py --im examples/vol_im.png --bgr examples/vol_bgr.png
python optimize.py --im examples/aerobie_im.png --bgr examples/aerobie_bgr.png
python optimize.py --im examples/pen_im.png --bgr examples/pen_bgr.png
```

The results will be written to the output folder.

Reference
------------
![Examples](examples/imgs/sfb_method.png)
If you use this repository, please cite the following [publication](https://arxiv.org/abs/2012.00595):

```bibtex
@inproceedings{sfb,
  title = {Shape from Blur: Recovering Textured 3D Shape and Motion of Fast Moving Objects},
  author = {Denys Rozumnyi and Martin R. Oswald and Vittorio Ferrari and Marc Pollefeys},
  booktitle = {NeurIPS},
  month = {Dec},
  year = {2021}
}
```
