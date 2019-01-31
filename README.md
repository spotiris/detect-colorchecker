# detect-colorchecker
A simple CNN (MobileNet V2) plus weights that can detect the Macbeth ColorChecker board. An application is included.


# Installation
  1. Clone this repo.
     ```bash
     git clone https://github.com/spotiris/detect-colorchecker.git
     ```
  2. Download most recent weights and model from [my Google Drive](https://drive.google.com/drive/folders/11a3E-iGnK58EzHz_RDyqr-pu12q9jsgf?usp=sharing) and put them in the repo directory.
  3. Have some images prepared and open a terminal in the repo directory. 
     See the help: `python3 detect-colorchecker.py --help`
  4. Feed the application `detect-colorchecker.py` file paths through STDIN, they can be either absolute or relatively referenced. Change 
     ```bash
     ls -1 rgb_small_jpg/left/*.jpg |
     CUDA_VISIBLE_DEVICES=0 stdbuf -i0 -o0 python3 detect-colorchecker.py . --batch-size 50
     ```
    
