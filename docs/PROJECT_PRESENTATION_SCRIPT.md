# 10-Minute Presentation Script: Opportunistic CT Lumbar Spine Segmentation Project

## Presentation Goal

This 10-minute script presents the project background, the idea of opportunistic CT, the segmentation principle, how the L1 trabecular bone core is generated, and the complete project workflow.

---

## 0:00-0:45 | Opening

Good morning everyone. Today I will present our project: an automated lumbar spine CT segmentation pipeline for extracting vertebral Hounsfield Unit measurements from routine abdominal CT scans.

The key idea is simple: many patients already undergo CT scans for clinical reasons unrelated to bone health. Instead of treating those scans as single-purpose images, we can reuse them opportunistically to assess bone quality, especially in the lumbar spine.

Our project focuses on segmenting lumbar vertebrae from CT DICOM images, generating a trabecular bone core from the L1 vertebral body, calculating Hounsfield Unit statistics, and exporting the results for comparison with DEXA bone mineral density measurements.

---

## 0:45-2:00 | Background: Osteoporosis And CT-Based Bone Assessment

Osteoporosis is a common skeletal disease characterized by reduced bone strength and increased fracture risk. Traditionally, bone mineral density is assessed using DEXA, or dual-energy X-ray absorptiometry. DEXA is widely used, low dose, and standardized, but it also has limitations.

For example, DEXA is a two-dimensional projection measurement. It can be affected by degenerative spine changes, vascular calcifications, scoliosis, body size, and positioning. In some patients, especially older adults, DEXA may overestimate lumbar spine bone density because it includes dense cortical bone, osteophytes, and other non-trabecular structures.

CT, on the other hand, contains three-dimensional anatomical information and voxel-level Hounsfield Unit values. Hounsfield Units reflect tissue attenuation. In bone, higher HU generally indicates denser mineralized tissue, while lower HU may suggest reduced bone density.

This creates an opportunity: if a patient already has a CT scan, we may be able to extract useful bone quality information without performing an additional scan.

---

## 2:00-3:00 | What Is Opportunistic CT?

Opportunistic CT means using CT scans that were acquired for another clinical purpose to extract additional clinically meaningful information.

For example, an abdominal CT may have been ordered for abdominal pain, cancer staging, infection, trauma, or surgical planning. Even though the scan was not ordered to assess osteoporosis, it often includes the lumbar vertebrae, especially L1 to L5.

So instead of asking, "Was this CT ordered for bone density?", opportunistic CT asks, "Can we reuse the existing CT data to estimate bone status?"

The advantages are important:

- No additional radiation exposure, because the image already exists.
- No extra scan time for the patient.
- Potential large-scale retrospective analysis from existing imaging archives.
- Better three-dimensional localization of trabecular bone compared with projection-based measurements.

In this project, the opportunistic CT target is the lumbar spine. We extract vertebral regions from abdominal CT images and calculate HU statistics, which can then be correlated with DEXA BMD values.

---

## 3:00-4:30 | Segmentation Principles

The central technical task is segmentation. Segmentation means assigning image voxels to anatomical structures. In our case, we want to identify lumbar vertebrae, especially L1 to L5, from a CT volume.

The input CT begins as a DICOM series. Each DICOM file is one CT slice, and the complete series forms a three-dimensional volume. The pipeline converts this DICOM series into a NIfTI image, preserving voxel spacing and orientation information.

For anatomical segmentation, the project uses TotalSegmentator. TotalSegmentator is a deep learning model trained to segment many anatomical structures in CT images. We use it in a focused way: the pipeline requests the lumbar vertebrae labels, including L1, L2, L3, L4, and L5.

In simple terms, many medical segmentation models are based on a U-Net-like idea. A U-Net first compresses the image to understand the larger anatomical context, then expands it back to the original image size to make voxel-by-voxel predictions. The "U" shape comes from this down-and-up structure. The down-sampling path helps the model recognize what structure it is looking at, while the up-sampling path helps it draw the boundary in the correct location.

For this project, the important point is not the exact neural-network architecture, but the output: for each vertebra, the model produces a mask showing which voxels belong to that anatomy.

The principle is:

1. The CT volume is passed into the segmentation model.
2. The model predicts binary masks for specific vertebrae.
3. Each mask has the same spatial relationship to the CT volume.
4. Voxels inside a mask are treated as belonging to that vertebral structure.
5. HU values are sampled from the CT image within each mask.

The project also includes quality-control visualization. The segmentation masks are overlaid on the original CT scan, so the user can verify whether the colored labels correctly align with the vertebral bodies. This is important because automated segmentation should not be blindly trusted, especially when the result may be used for research or clinical correlation.

---

## 4:30-6:00 | Getting The Trabecular Bone Core

A major feature of this project is generating a trabecular core mask for the L1 vertebral body.

We start with L1 because it is commonly visible on routine abdominal CT scans and is often less affected by severe lower-lumbar degenerative change than L4 or L5. It is also a practical reference level for opportunistic osteoporosis assessment because many published CT HU studies use L1 as a representative vertebral measurement. In short, L1 is usually available, anatomically consistent, and clinically meaningful for comparison with bone-density data.

Why do we need this?

The vertebra contains both cortical bone and trabecular bone. Cortical bone is the dense outer shell. Trabecular bone is the inner cancellous structure and is more metabolically active. For osteoporosis assessment, trabecular bone is often more informative because it is affected earlier by bone loss.

If we measure the entire vertebra, the result may include cortical shell, posterior elements, endplates, and other dense structures. These can raise the HU value and make the measurement less specific to cancellous bone quality.

So the project first generates a vertebral body mask and then extracts an inner core.

The process is:

1. TotalSegmentator generates whole vertebra masks, such as `vertebrae_L1.nii.gz`.
2. It also generates a general vertebral body mask, `vertebrae_body.nii.gz`.
3. The pipeline intersects the L1 vertebra mask with the vertebral body mask.
4. This produces `vertebrae_L1_body.nii.gz`, which isolates the L1 vertebral body instead of the entire vertebra.
5. The L1 body mask is then eroded inward by 2.5 mm.
6. Voxels within 2.5 mm of the mask boundary are removed.
7. The remaining inner region is saved as `vertebrae_L1_body_trabecular_core.nii.gz`.

Technically, the erosion is done using a distance transform. For every voxel inside the L1 body mask, the algorithm calculates its distance to the nearest background voxel. Only voxels more than 2.5 mm away from the boundary are kept.

This is better than simply removing a fixed number of pixels, because CT voxel spacing can vary between scans. A 2.5 mm erosion is a physical distance, not just an image-index distance.

The result is a more standardized inner trabecular region for HU measurement.

---

## 6:00-8:30 | Project Workflow

Now I will walk through the complete workflow.

First, the user provides an input folder containing DICOM CT scans. The project supports both a single patient folder and a batch folder containing multiple patients or studies.

Second, the pipeline searches for DICOM files and groups them into cases. Patient IDs are extracted from DICOM metadata when available. If metadata is missing, the folder name can be used as a fallback. Patient IDs are normalized so they can later be matched with external data such as DEXA records.

Third, the DICOM series is converted into a NIfTI image. During this step, the pipeline applies CT rescale slope and intercept so voxel intensities are represented as Hounsfield Units. It also builds an affine matrix from DICOM orientation and spacing information, then converts the image into a canonical RAS orientation.

Fourth, TotalSegmentator is run. The pipeline requests lumbar vertebra segmentation and generates masks for the vertebral levels. It also attempts to generate the vertebral body mask so that labeled vertebral bodies and the L1 trabecular core can be derived.

Fifth, the pipeline computes measurements. For each available mask, it counts the mask voxels, calculates volume using voxel spacing, and extracts the CT intensities inside the mask. The main statistic is the mean HU value within the segmented region.

Sixth, the project saves outputs. For each patient, it creates a patient-specific output folder containing:

- Segmentation masks in NIfTI format.
- A statistics JSON file.
- A statistics CSV file.
- Preview images showing segmentation overlays.

For batch processing, the project also creates a consolidated CSV file, `batch_statistics.csv`, which can be used for statistical analysis and DEXA matching.

Seventh, the user performs verification. The verification tools overlay color-coded segmentation masks on axial and sagittal CT views. The user can inspect whether L1 through L5 are correctly labeled, whether the masks align with bone boundaries, and whether there are orientation problems.

This verification step is especially important because CT scans can vary in orientation, field of view, reconstruction kernel, slice thickness, and image quality.

---

## 8:30-9:30 | Why This Workflow Is Useful

The value of this project is that it turns routine CT scans into structured quantitative data.

Instead of manually drawing regions of interest, which is time-consuming and operator-dependent, the pipeline automates the segmentation and measurement process. Instead of measuring arbitrary slices, it uses anatomical masks. Instead of only outputting images, it exports structured CSV files that can be matched with DEXA BMD values.

The L1 trabecular core is particularly useful because it focuses the measurement on cancellous bone and avoids much of the cortical shell. This may provide a cleaner CT-based signal for osteoporosis-related analysis.

The project is also designed for practical batch workflows. It can process many patients, skip already processed cases, generate outputs patient by patient, and support later visual quality control.

---

## 9:30-10:00 | Closing

In summary, this project builds an automated pipeline for opportunistic CT-based lumbar bone assessment.

It takes routine abdominal CT DICOM images, segments lumbar vertebrae using TotalSegmentator, derives the L1 trabecular bone core using a 2.5 mm distance-based erosion, calculates HU intensity statistics, and exports results for verification and DEXA correlation.

The broader goal is to make better use of imaging data that already exists. By extracting bone information from routine CT scans, opportunistic CT may help identify patients at risk of osteoporosis without requiring additional imaging.

Thank you.

---

## Optional Slide Outline

1. Project title and motivation
2. Osteoporosis and limitations of DEXA
3. Definition of opportunistic CT
4. CT HU values and bone density concept
5. Segmentation principle using TotalSegmentator
6. L1 vertebral body and trabecular core generation
7. End-to-end workflow diagram
8. Outputs: masks, JSON, CSV, preview, QC
9. DEXA matching and research use
10. Summary and future potential
