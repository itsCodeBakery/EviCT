DESCRIPTION

This is the dataset used in KISEG: A Three-Stage Segmentation Framework for Multi-level Acceleration of Chest CT Scans from COVID-19 Patients.

1. Folder structure:
    image: 150 CT scans indexed from 0 to 149. Each CT scan folder contains frame jpg files indexed in a serial order.
    mask: 150 CT scans indexed from 0 to 149. Each mask folder contains annotatation png files for the frame jpg file with the same filename and the same CT scan index.

2. Annotation and image file explanation:
    frame jpg files and the corresponding mask png files are all with a resolution of 512x512.
    Each pixel of a mask png file is a uint8 number ranged from 0 to 3 represents the pixel-wised label for the corresponding frame jpg file.
    The meaning of label from 0 to 3 are as follow:
        0: Background (BG)
        1: Lung field (LF)
        2: Ground-glass opacity (GGO)
        3: Consolidation (CO)
