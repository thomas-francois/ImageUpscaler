# ImageUpscaler

This project is a simple repackage of the [Real-ESRGAN model](https://github.com/xinntao/Real-ESRGAN) to be simpler to install, use and debug on MacOS.


## Requirements :
- Python 3.12.2
- pytorch
- numpy
- opencv-python

## Installation

1. Insatll repo :
    - With git :
      ```bash
      git clone https://github.com/thomas-francois/ImageUpscaler/
      cd ImageUpscaler
      ```
    
    
    - Direct download :<br/><img width="450" alt="DownloadPreview" src="https://github.com/user-attachments/assets/251eb907-952b-4eee-934a-10223548e015"><br/>
      - Uncompress ImageUpscaler.zip
      - Open folder ImageUpscaler-main in terminal (Right click the folder > services)

2. Install requirements :  
      - Run in terminal : ```sh setup.sh```


## Usage

To upscale your image, simply run in the terminal (from the folder ImageUpscaler) the command :  
```bash
venv/bin/python Upscale.py
```

> [!NOTE]  
> **By default** the program will upscale all images from the ```inputs```folder by a factor of ```4``` and the results will be placed in the ```results``` folder

<br/>

### Options
You can configure the command with this options :

```console
  -i inputPath            input image path (jpg/png/webp) or directory
  -o outputPath           output image path (jpg/png/webp) or directory
  -s scale                upscale ratio (default: 4)
  -n speed                model speed (and conversely quality) (choices: fast / medium / hight, default: fast)
  --suffix outputSuffix   suffix to append to upscaled image (default: the model speed)
```

---

**Example of custom command :**  
```bash
.venv/bin/python Upscale.py -i /Users/me/Downloads/dog.png -o /Users/me/Downloads -s 2 -n hight
```  
This command will upscale the image **dog.png** by a factor of **2** with the hight quality model and place the upscaled image in the **Downloads** folder  

> [!Tip]
> You can drap and drop files and folders in the terminal instead of typing their path.
