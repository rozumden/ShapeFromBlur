# ShapeFromBlur
Shape from Blur: Recovering Textured 3D Shape and Motion of Fast Moving Objects

Make sure to download the official open-source DeFMO implementation from GitHub: https://github.com/rozumden/DeFMO
e.g. by 'git clone https://github.com/rozumden/DeFMO.git'
Please, also set up the DeFMO pre-trained models in main_settings.py by setting the variable g_saved_models_folder

Kaolin is available here: https://github.com/NVIDIAGameWorks/kaolin

The code can be easily run by 'python optimize.py' 
or with your inputs as:
python optimize.py --im examples/vol_im.png --bgr examples/vol_bgr.png
python optimize.py --im examples/aerobie_im.png --bgr examples/aerobie_bgr.png
python optimize.py --im examples/pen_im.png --bgr examples/pen_bgr.png

The results will be written to the output folder.