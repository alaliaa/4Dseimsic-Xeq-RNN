# 4Dseimsic-Xeq-RNN

This is the code for the paper [Time-lapse data matching using a recurrent neural network approach](https://library.seg.org/doi/epub/10.1190/geo2021-0487.1) ([arXive version](https://arxiv.org/abs/2204.00941))

**Abstract:**
*Time-lapse seismic data acquisition is an essential tool to monitor changes in a reservoir due to fluid injection, such as CO2 injection. By acquiring multiple seismic surveys in the exact same location, the authors can identify the reservoir changes by analyzing the difference in the data. However, such analysis can be skewed by the near-surface seasonal velocity variations, inaccuracy, and repeatability in the acquisition parameters, and other inevitable noise. The common practice (cross equalization) to address this problem uses the part of the data in which changes are not expected to design a matching filter and then apply it to the whole data, including the reservoir area. Like cross equalization, the authors train a recurrent neural network (RNN) on parts of the data excluding the reservoir area and then infer the reservoir-related data. The RNN can learn the time dependency of the data, unlike the matching filter that processes the data based on the local information obtained in the filter window. The authors determine the method of matching the data in various examples and compare it with the conventional matching filter. Specifically, they start by demonstrating the ability of the approach in matching two traces and then test the method on a prestack 2D synthetic data. Then, the authors verify the enhancements of the 4D signal by providing reverse time migration images. The authors measure the repeatability using normalized root-mean-square and predictability metrics and find that, in some cases, their proposed method performed better than the matching filter approach.*

# Table of contents 
The folders correspond to the following experiments in the paper:


**:open_file_folder:  two_traces:**  Matching two traces

**:open_file_folder:  Otway_20samples:** Otway model 
    
**:open_file_folder:  SEAM_20sample:** SEAM TL model

**:open_file_folder:  Noise_Otway_20samples:** Application on noisy data



 # Data 
https://drive.google.com/drive/folders/1jx2wnGJOyY1m7pKpdotH1Yy86CPirWfr?usp=sharing 

 
