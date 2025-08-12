# Trab_analysis_microct

This repository contains the code for training, evaluation, and inference of a deep learning–based model for classifying cross-sectional slices from 3D micro-CT images of mouse tibiae into four anatomical compartments:
- Epiphyseal bone
- Growth plate
- Primary spongiosa
- Secondary spongiosa

This classification enables the detection of transitional interfaces between: the epiphyseal bone and the growth plate ($Z_{eg}$), the growth plate and the primary spongiosa ($Z_{gp}$), and the primary and secondary spongiosa ($Z_{ps}$).

Identifying these landmarks allows for the extraction of consistent volumes of interest (VOIs) across different experimental groups and study datasets. These standardized VOIs are then used in subsequent steps of the analysis pipeline to segment trabecular bone within each compartment, followed by morphological and statistical analyses. This approach enables a comprehensive and reproducible assessment of trabecular bone in mouse tibiae.

Future work will involve publishing a paper presenting the complete automated end-to-end pipeline for the analysis of cortical and trabecular bone from micro-CT scans of rodent models. This pipeline will include:
- Data structure standardization
- Preprocessing
- Pre-alignment and registration
- Trabecular VOI extraction
- Trabecular compartment segmentation
- Morphological and statistical analysis of cortical and trabecular bone

Once published, this repository will be linked to the main, final pipeline repository [SOON].
## TODO
1. Add documentation on code functionality and dataset structure.
