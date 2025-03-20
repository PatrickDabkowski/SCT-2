import os
import SimpleITK as sitk
from lungmask import LMInferer

def read_ct(path: str):
    """
    reads CT as DICOM series | .mha | .nrrd
    Args:
        path (str): path to CT
    Returns: 
        image (sitk.Image): extracted CT
        name (str): sample name
    """
    
    if os.path.isdir(path): 
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        name = path.split("/")[-1]
        print("DICOM Series read")
    
    elif path.endswith((".mha", ".nrrd")): 
        image = sitk.ReadImage(path)
        if path.endswith((".mha")):
            name = path.split(".mha")[-2]
            name = name.split("/")[-1]
        elif  path.endswith((".nrrd")):
            name = path.split(".nrrd")[-2]
            name = name.split("/")[-1]
            
        print(f"File {path} read")
    
    else:
        raise ValueError("Unknown format. Supported: DICOM (folder), .mha, .nrrd")

    return image, name

def segment_lungs(path: str):
    """
    segments lungs from CT and creates their files in /temporary_data/sample_name
    Args:
        path (str): path to CT
    """
    
    inferer = LMInferer()

    input_image, name = read_ct(path)
    
    os.mkdir(f"temporary_data/{name}")
    sitk.WriteImage(input_image, f"temporary_data/{name}/ct.nrrd")

    segmentation = inferer.apply(input_image)  # default model is U-net(R231)
    segmentation = sitk.GetImageFromArray(segmentation)
    segmentation.CopyInformation(input_image)

    binary_mask = sitk.BinaryThreshold(segmentation, lowerThreshold=1, upperThreshold=2, insideValue=1, outsideValue=0)

    sitk.WriteImage(binary_mask, f"temporary_data/{name}/lungs.nrrd")