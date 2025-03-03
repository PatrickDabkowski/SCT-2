import os
import torch
import argparse
import numpy as np
import SimpleITK as sitk
from torchvision import transforms
from collections.abc import Sequence
from models import Encoder, GRUAggregation, MLP

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((512, 512), antialias=True), 
                                    transforms.ConvertImageDtype(torch.float32)])

def preprocessing(img: np.ndarray):
    """
    Clips CT slice into working range
    Args:
        img (np.ndarray): CT slice
    """
    img = np.clip(img, a_min=-1000, a_max=1000)

    min = -1000
    max = 1000

    return (img - min) / (max - min)

def check_lungs(slice: np.ndarray):
    """
    check weather lungs are more than 6% of sample
    Args:
        slice (np.ndarray): CT slice
    Returns: 
        (Bool): True or False wether lungs are more than 6% of slice
    """
    count_diff = np.count_nonzero((slice != np.min(slice))) 
    
    total_elements = slice.shape[0] * slice.shape[1]
    percent_diff = (count_diff / total_elements) * 100
    
    return percent_diff > 6

class CT2Inference():
    
    def __init__(self, threshold: float = 0.15, from_pretrained: Sequence[str]|None = None, device: str=None):
        """
        Initialize CT^2 object for inferece 
        Args:
            threshold (float): decision boundary for classifier
            device (str): computing node (cuda, mps, cpu)
            from_pretrained (Sequence[str]): list of paths to model weights
        """
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("device: cuda (NVIDIA GPU)")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("device: mps (Mac GPU)")
            else:
                self.device = torch.device("cpu")
                print("device: cpu")
        
        self.threshold = threshold

        self.encoder = Encoder()
        self.aggregation = GRUAggregation(512, 2)
        self.classifier = MLP(1)
        
        if from_pretrained:

            self.encoder.load_state_dict(torch.load(from_pretrained[0]))
            self.aggregation.load_state_dict(torch.load(from_pretrained[1]))
            self.classifier.load_state_dict(torch.load(from_pretrained[2]))

        self.encoder.to(self.device)
        self.aggregation.to(self.device)
        self.classifier.to(self.device)
        
    def process_slice(self, slice: np.ndarray):
        """
        process single CT slice by Encoder to achive slice embedding
        Args:
            slice (np.ndarray): CT slice
        Returns: 
            latent_space (np.ndarray): latent features extracted by Encoder
        """
        img = preprocessing(slice)  
             
        # use autoencoder                           
        img = transform(img).unsqueeze(0).to(self.device)
    
        latent_space = self.encoder(img)
        
        return latent_space.squeeze().cpu().detach().numpy()
    
    def process_sample(self, sample_path):
        """
        process whole CT by aggregation of slice embeddings
        Args:
            slice (np.ndarray): CT slice
        Returns: 
            values (torch.Tensor): aggregated CT representation
        """
        sample_path = sample_path.split('\\')
        lungs = sitk.GetArrayFromImage(sitk.ReadImage(sample_path[0] + "\\Lungs\\" + sample_path[-1]))
        study = sitk.GetArrayFromImage(sitk.ReadImage(sample_path[0] + "\\Studies\\" + sample_path[-1]))
        sample = study.copy()
        sample[lungs==0] = np.min(sample)
        
        bag = []
        
        for i in range(sample.shape[0]):
            
            #check weather lungs are more than 6% of sample
            if check_lungs(sample[i]):
                latent_space = self.process_slice(sample[i], i)
                bag.append(latent_space)
            
        # aggregation of slice embeddings
        bag = torch.tensor(np.array(bag)).to(self.device)
    
        values, weights = self.aggregation(bag)
            
        return values 
    
    @torch.no_grad()
    def __call__(self, sample_path: str, lungs_path: str):
        """
        classifies CT based on their embeddings
        Args:
            sample_path (str): path to CT's directory 
            lungs_path (str): path to CT's segmentation
        Returns: 
            predictions (torch.Tensor[int]): binary classification of the cancer
            probabilities (torch.Tensor[float]): class probabilities
        """
        self.encoder.eval()
        self.aggregation.eval()
        self.classifier.eval()


        sample_path = sample.split('\\')
        if os.path.exists(sample_path[0] + "\\Lungs\\" + sample_path[-1]) and os.path.exists(sample_path[0] + "\\Studies\\" + sample_path[-1]):
            bag_space = self.process_sample(sample)
        
            pred = self.classifier(bag_space)
            probabilities = torch.sigmoid(pred)
            predictions = (probabilities >= self.threshold).float() 
        
        return predictions, probabilities

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run (S-CT)^2 Inference")
    parser.add_argument("--input", type=str, help="path to input file") 
    parser.add_argument("--lungs", type=str, help="path to CT's segmentation")
    parser.add_argument("--threshold", type=float, help="decision boundary for classifier", default=0.15)
    parser.add_argument("--device", type=str, help="computing node (cuda, mps, cpu)") 
    parser.add_argument("--from_pretrained", type=Sequence([str]), help="list of paths to model weights", default=["model_weights/_encoder.pt", "model_weights/_aggregation.pt", "model_weights/_ct.pt"]) 
    
    args = parser.parse_args()
    
    sct2 = CT2Inference(threshold=args.threshold, device=args.device, from_pretrained=args.from_pretrained)
    
    if args.lungs is None:
        pass
    
    predictions, probabilitiessct2 = (args.input)