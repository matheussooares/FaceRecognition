import imageio.v2 as imageio
import cv2
from numpy import uint8, zeros, stack

def tranformImagem(dict_file, extensoes: list = [".png", ".jpg", ".jpeg", ".bmp", ".pgm"], size:tuple = (20,20), flatten: bool = True):
    # Lista de paths de imagens
    imagens_paths = [str(p) for p in dict_file.glob("*") if p.suffix.lower() in extensoes]
    if flatten:
        # Construi a base de dados
        X = zeros((len(imagens_paths), size[0]* size[1]), dtype=uint8)
    else:
        X = zeros((len(imagens_paths), size[0], size[1]), dtype=uint8)
    
    for i, img_path in enumerate(imagens_paths):
        # abre a imagem
        img = imageio.imread(img_path)   
        # Redimensionar a imagem
        img = cv2.resize(img, (size[0], size[1]))
        if flatten:
            # Salva o vetor da imagem
            X[i, :] = img.flatten()
        else:
            # X = stack([X, img])
            X[i,:,:] = img
    
    return X