# Research Funding Proposal: AI Workstation For Opportunistic CT-Based Osteoporosis Screening

## Project Title

AI-Assisted Opportunistic Osteoporosis Screening From Routine Abdominal CT Using Automated Lumbar Spine Segmentation And Hounsfield Unit Analysis

## Principal Research Aim

This project seeks funding to establish a dedicated artificial intelligence workstation for the development, validation, and deployment of an automated CT-based lumbar spine analysis pipeline. The workstation will support high-throughput medical image processing, deep learning segmentation, quality-control visualization, and statistical correlation of CT-derived Hounsfield Unit (HU) measurements with dual-energy X-ray absorptiometry (DEXA) bone mineral density (BMD) results.

The proposed infrastructure will support both the initial validation work and future research studies that build on the same pipeline. Retrospective CT datasets can be processed locally, safely, and reproducibly to establish evidence and operating procedures. The same workstation can later be used for new patient cohorts, protocol refinements, and follow-on studies without procuring a separate compute platform. Imaging data are expected to flow in and out of the workstation for processing rather than being retained long term on the machine. Long-term storage remains on existing institutional systems.

Beyond research efficiency, local infrastructure creates a pathway toward patient benefit. If validated, opportunistic CT assessment could help identify bone-health risk in patients who already have routine CT imaging but may not have undergone DEXA screening, without requiring additional radiation exposure or scan appointments.

## Executive Summary

Osteoporosis is common, underdiagnosed, and clinically important because it increases fracture risk, morbidity, mortality, and healthcare cost. DEXA remains the reference standard for BMD assessment, but many patients at risk for osteoporosis do not undergo DEXA screening. At the same time, large numbers of patients already receive abdominal or body CT examinations for unrelated clinical indications, such as malignancy staging, abdominal pain, infection, trauma, and surgical planning.

Routine CT scans contain the lumbar spine and provide voxel-level HU information that reflects bone attenuation. This creates an opportunity to reuse existing CT imaging for additional bone-health assessment without exposing patients to additional radiation, requiring extra scan time, or adding new imaging appointments.

The current project implements an automated lumbar spine CT segmentation pipeline. It processes DICOM CT series, segments lumbar vertebrae using TotalSegmentator, derives an L1 trabecular bone core through distance-based erosion, calculates HU statistics, generates visual overlays for quality control, and exports structured results for DEXA matching and statistical analysis.

Funding is requested for a dedicated AI workstation to enable local, institution-controlled processing of computationally demanding 3D CT volumes, deep learning segmentation, PyTorch and nnU-Net workloads, and unattended batch analysis. A local service model is expected to be more cost-effective over the full research and translation lifecycle than recurring vendor-hosted or cloud-based alternatives, while keeping sensitive imaging data under institutional governance. Current lower-memory systems can run limited inference, but they are not adequate for reliable large-scale CT batch processing, quality-control visualization, and reproducible analysis across initial validation cohorts and future studies.

## Background And Significance

Osteoporosis is often clinically silent until a fragility fracture occurs. Hip, vertebral, and wrist fractures can lead to chronic pain, disability, loss of independence, and increased mortality. Early detection is therefore central to preventive care.

DEXA is widely used for measuring BMD, but it has practical and technical limitations. Access may be limited, screening rates may be low, and lumbar spine measurements can be affected by degenerative change, vascular calcification, scoliosis, body habitus, positioning, and projection-related artifacts. CT imaging, in contrast, provides three-dimensional anatomical information and direct attenuation values in HU.

Opportunistic CT screening uses imaging that has already been acquired for other clinical reasons to extract additional clinically relevant information. For osteoporosis research, this means identifying vertebral regions on routine CT and measuring bone attenuation in standardized locations. The L1 vertebral body is particularly useful because it is frequently included on abdominal CT, is commonly used in published HU-based osteoporosis studies, and can be compared with DEXA measurements.

Manual CT region-of-interest measurement is time-consuming and operator-dependent. Automated segmentation offers a scalable solution. By using deep learning segmentation to identify vertebral anatomy, the project can generate reproducible HU measurements across large datasets and support research into CT-derived osteoporosis screening thresholds.

## Research Problem

The research problem is not merely whether CT HU values correlate with DEXA BMD. Prior studies suggest that such a relationship exists. The practical research challenge is how to build a reproducible, scalable, and quality-controlled pipeline that can process real-world institutional CT data at sufficient volume for meaningful local validation.

The project addresses several barriers:

- Routine CT studies vary in slice thickness, field of view, reconstruction kernel, orientation, and image quality.
- Large clinical and research CT datasets require automated batch processing rather than manual measurement.
- Deep learning segmentation of 3D CT volumes requires substantial GPU memory and system RAM.
- Research outputs must be auditable, exportable, and suitable for statistical analysis.
- Automated results require visual quality control before being used for clinical research conclusions.

## Specific Objectives

1. Develop and validate an automated pipeline for extracting lumbar vertebral HU measurements from routine abdominal CT DICOM studies.
2. Generate standardized L1 trabecular core measurements by combining vertebral segmentation with vertebral body localization and distance-based erosion.
3. Correlate CT-derived HU measurements with matched DEXA BMD and T-score values.
4. Evaluate segmentation success rate, failure modes, and quality-control requirements across heterogeneous clinical CT studies.
5. Establish a local AI workstation environment capable of secure, reproducible, high-throughput medical image processing with minimal on-device data retention.
6. Create a reusable infrastructure foundation for future musculoskeletal AI research, including validation and deployment of additional opportunistic CT biomarkers.
7. Prepare a sustainable local pathway from initial research validation toward future studies that may improve patient identification, referral, and bone-health assessment using routine CT already performed for other clinical reasons.

## Research Questions

The project will address the following questions:

- Can an automated segmentation pipeline reliably extract L1-L5 vertebral HU measurements from routine abdominal CT studies?
- Does the L1 trabecular core HU measurement correlate with DEXA-derived lumbar spine or hip BMD?
- What HU threshold or prediction model best identifies patients with low BMD or osteoporosis in the local population?
- What proportion of routine CT studies are suitable for automated opportunistic osteoporosis assessment?
- Which imaging factors most commonly cause segmentation failure or unreliable HU measurement?
- How much computational capacity is required to process CT datasets in a practical timeframe for both initial validation and future studies?

## Innovation

This project is innovative because it converts existing routine CT data into structured bone-health measurements using an automated AI-assisted workflow. Instead of depending on manual region-of-interest placement, the pipeline uses anatomical segmentation to define measurement regions consistently. Instead of measuring only a single 2D slice, the workflow can calculate volumetric HU statistics from segmented 3D vertebral masks.

The L1 trabecular core workflow is particularly important. The project isolates the L1 vertebral body and erodes the mask inward by a physical distance of 2.5 mm. This approach reduces contamination from cortical bone, endplates, posterior elements, and boundary partial-volume effects. Because the erosion is distance-based rather than pixel-count-based, it is better suited to CT studies with variable voxel spacing.

The proposed workstation also creates a platform for future medical AI work. Once the CT and DEXA dataset has been curated, the same infrastructure can support threshold analysis, regression modeling, classification models, segmentation refinement, and additional opportunistic screening tasks.

## Current Technical Foundation

The existing software pipeline already includes the following capabilities:

- DICOM folder discovery and patient-level batch processing.
- DICOM metadata extraction and patient identifier normalization for research matching.
- Conversion from DICOM series to NIfTI volume format.
- Automated lumbar vertebra segmentation using TotalSegmentator.
- Generation of L1 vertebral body and L1 trabecular core masks.
- HU intensity measurement and volume calculation.
- Per-patient structured statistics output.
- Consolidated batch outputs for statistical analysis.
- Preview image generation with segmentation overlays.
- Interactive verification workflow for marking segmentation success or failure.
- Windows-compatible batch graphical user interface for processing and quality control.

The requested workstation is therefore not for speculative software development alone. It will accelerate and stabilize an active research pipeline that already has a defined clinical use case and a working computational workflow.

## Proposed Study Design

### Study Scope

The workstation supports a research program with two connected phases:

1. **Initial validation phase:** A retrospective observational imaging study to validate the automated pipeline, correlate CT HU measurements with DEXA, and define quality-control requirements in real-world institutional data.
2. **Future study phase:** Prospective or operational research studies that apply the validated workflow to new patient cohorts, refined protocols, and potential clinical translation pathways.

This design recognizes that the infrastructure is not intended for a single archived dataset alone. It is intended to remain useful as the institution moves from initial evidence generation toward future studies that may deliver added benefit to patients.

### Initial Validation Study Type

Retrospective observational imaging study.

### Initial Validation Study Population

The initial validation cohort will include adult patients who have undergone abdominal, thoracoabdominal, or body CT imaging that includes the lumbar spine and who have a DEXA examination within a clinically acceptable time window.

The final inclusion and exclusion criteria will be refined according to institutional review board requirements and data availability.

### Proposed Inclusion Criteria

- Adult patients with routine CT imaging that includes L1 and at least part of the lumbar spine.
- Available DEXA BMD result within the defined matching interval.
- CT images available in DICOM format.
- CT image quality sufficient for vertebral segmentation and HU analysis.

### Proposed Exclusion Criteria

- Severe motion artifact or incomplete lumbar spine coverage.
- Prior spinal instrumentation or extensive vertebral hardware affecting the measured level.
- Vertebral fracture, destructive lesion, or severe deformity at the target measurement level when it invalidates HU analysis.
- CT series with nonstandard reconstruction that prevents reliable HU measurement.
- Missing or inconsistent identifiers preventing safe matching of CT and DEXA data.

### Data Elements

The study will collect or derive:

- CT study date, series metadata, slice thickness, reconstruction kernel, and scanner information where available.
- Segmentation masks for lumbar vertebrae L1-L5.
- L1 vertebral body mask and L1 trabecular core mask.
- Mean, median, standard deviation, and volume-based HU statistics.
- DEXA BMD, T-score, Z-score, and diagnosis category where available.
- Quality-control status for each processed case.
- Failure reason for cases rejected during review.

## Methodology

### 1. Data Preparation

Eligible CT studies will be exported or accessed in DICOM format according to institutional data governance rules. The project will organize cases by study or patient identifier and record metadata needed for matching and auditability. Identifiers used for analysis will be managed according to approved research protocol requirements.

### 2. Automated Segmentation

The pipeline will process each CT series through an automated segmentation workflow. TotalSegmentator will be used to generate lumbar vertebral masks, including L1-L5. The model runs on PyTorch and nnU-Net components and benefits substantially from GPU acceleration, especially when processing full 3D CT volumes.

### 3. L1 Trabecular Core Extraction

The L1 vertebral mask will be combined with a vertebral body mask to isolate the L1 vertebral body. A distance transform will then erode the body mask inward by 2.5 mm to generate an inner trabecular core. This physical-distance erosion is preferred because it accounts for variation in CT voxel spacing.

### 4. HU Measurement

The pipeline will sample CT voxels within each segmentation mask and calculate HU statistics. The main research measurement will be the L1 trabecular core mean HU. Secondary measurements may include whole L1, L2-L5 vertebrae, multi-level averages, and volume-based metrics.

### 5. Quality Control

Automated outputs will be reviewed using the existing overlay viewer. Reviewers will mark cases as successful, failed, or not applicable. Quality-control results will be recorded systematically, allowing exclusion of unreliable measurements before statistical analysis.

### 6. DEXA Matching

CT-derived outputs will be matched with DEXA measurements using approved identifiers and date windows. Analysis datasets will include CT HU metrics, DEXA BMD values, T-scores, and relevant metadata.

### 7. Statistical Analysis

The analysis will include:

- Descriptive statistics for patient cohort and imaging characteristics.
- Correlation between CT HU values and DEXA BMD.
- Comparison of HU values across normal BMD, osteopenia, and osteoporosis groups.
- Receiver operating characteristic analysis for detecting low BMD or osteoporosis.
- Regression models evaluating the relationship between CT HU and DEXA values.
- Sensitivity analysis by CT protocol, slice thickness, reconstruction kernel, and scanner type where sample size permits.
- Inter-reviewer or intra-reviewer quality-control assessment if multiple reviewers are involved.

## Expected Outcomes

The project is expected to produce:

- A validated local pipeline for opportunistic CT-based lumbar bone assessment.
- A curated dataset of CT-derived lumbar HU measurements matched to DEXA results.
- Evidence describing the relationship between L1 trabecular HU and DEXA BMD in the local population.
- Practical knowledge of segmentation failure modes in real-world CT data.
- A documented batch workflow reusable for future retrospective, prospective, and quality-improvement imaging studies.
- A translational foundation for future patient-facing opportunistic screening research.
- Conference abstracts, manuscript submissions, or institutional research outputs.
- A reusable AI workstation environment for future medical imaging AI projects.

## Potential Benefits Of Local AI Infrastructure

Establishing an in-house AI workstation creates value beyond completing one initial validation study. The main benefits are patient-oriented, scientific, operational, and financial.

### Potential Patient Benefits

- **No additional imaging burden:** Opportunistic assessment uses CT already acquired for another clinical indication, avoiding extra radiation exposure and additional scan appointments when bone information can be extracted from existing studies.
- **Better use of routine care:** Patients who receive abdominal or body CT for unrelated reasons may gain additional bone-health information that would otherwise remain unused in the image.
- **Earlier risk awareness:** If future studies confirm useful thresholds, automated HU assessment could help identify patients with possible low bone density who have not yet undergone DEXA screening.
- **More consistent assessment:** Standardized vertebral segmentation and L1 trabecular core measurement may reduce variability compared with manual review of single slices.
- **Support for future clinical pathways:** Validated local infrastructure makes it more feasible to test referral, monitoring, or preventive-care workflows in later patient-focused studies.

These patient benefits depend on future validation and governance approval. The workstation funding request is for research infrastructure that makes those future studies possible.

### Scientific And Clinical Benefits

- **Scalable opportunistic screening:** Routine abdominal CT can be converted into structured lumbar HU measurements across large cohorts without manual region-of-interest placement.
- **Standardized measurement:** Automated segmentation and the L1 trabecular core workflow improve consistency compared with ad hoc slice-based measurement.
- **Faster research iteration:** The team can refine inclusion criteria, quality-control rules, and analysis pipelines without waiting for external service turnaround.
- **Local validation:** The institution can generate evidence specific to its CT protocols, scanners, and patient population rather than relying on published thresholds alone.
- **Reusable research platform:** The same infrastructure can support future musculoskeletal, body-composition, opportunistic CT biomarker, and patient-focused follow-on studies.

### Operational Benefits

- **Processing-in, processing-out workflow:** The workstation acts as a secure compute node while long-term imaging storage remains on existing institutional systems.
- **Integrated quality control:** Batch segmentation, visual verification, and case tracking can run in one controlled environment aligned with the current project software.
- **Reproducibility:** A fixed hardware and software stack makes it easier to repeat analyses, audit processing steps, and document methods for publication.
- **Independence from external processing queues:** Local batch jobs can run overnight and be resumed without dependency on third-party job scheduling or upload portals.

### Financial Benefits Compared With Vendor Or Cloud Solutions

Vendor-hosted AI services, commercial analysis platforms, and cloud GPU environments often appear attractive initially because they avoid upfront hardware procurement. For a research program that processes hundreds to thousands of CT studies across initial validation and future cohorts, performs repeated re-analysis during method development, and requires case-by-case quality control, recurring external costs usually accumulate quickly.

A local workstation is typically more cost-effective because:

- **Capital purchase versus recurring fees:** Hardware is a one-time research infrastructure investment, whereas vendor and cloud solutions commonly charge per study, per hour, per terabyte, or annual license fees.
- **No per-case upload and reprocessing penalty:** Research workflows frequently require re-running failed cases, testing protocol changes, and regenerating outputs. External services often charge again for each repeat.
- **Lower long-term cost at scale:** Once procured, the workstation can process additional cohorts, student projects, and follow-on grants without negotiating new service contracts.
- **Reduced hidden costs:** External solutions add data-transfer time, governance review for off-site processing, dependency on vendor roadmaps, and risk of workflow changes that break reproducibility.
- **Better alignment with open research software:** The project already uses an open, customizable Python pipeline. Local hosting preserves control over segmentation settings, trabecular core generation, DEXA matching, and export formats that may not be available in closed vendor products.

## Justification For AI Workstation Funding

### Why A Dedicated Workstation Is Required

This research requires a dedicated AI workstation because 3D CT segmentation and batch analysis are computationally intensive. A single CT study may contain hundreds to thousands of slices. Processing requires loading volumetric image arrays, running deep learning segmentation, generating multiple masks, saving NIfTI files, computing statistics, and creating verification outputs.

General-purpose office computers are not designed for this workload. On low-memory systems, segmentation may fail due to insufficient GPU memory, especially when the pipeline performs multiple TotalSegmentator passes per case and loads full 3D CT volumes. CPU-only processing is significantly slower and can make large-scale imaging research impractical. Shared services, commercial AI vendors, and cloud computing may introduce data governance, recurring cost, transfer delays, workflow inflexibility, and reproducibility concerns, especially when working with medical imaging data.

A local workstation allows the team to:

- Process protected imaging data within institutional control using a processing-in, processing-out workflow.
- Run GPU-accelerated segmentation reliably without per-study vendor processing fees.
- Complete batch processing within days rather than weeks or months.
- Avoid repeated cloud compute charges, subscription costs, and data-transfer barriers.
- Maintain a stable, reproducible software environment that is not tied to a vendor product roadmap.
- Support interactive quality control without competing for shared institutional or external service resources.
- Retain full control over the L1 trabecular core method, lumbar ROI selection, and DEXA-matching outputs required by this study.

### Workload Estimate

Current GPU inference for a CT case may require several minutes per patient depending on scan size, selected model mode, and hardware. CPU processing may require substantially longer and is more likely to encounter memory limitations. The workstation must therefore support unattended batch jobs, recover from individual case failures, and hold only the active cases and derived outputs needed for immediate processing.

The pipeline performs more than one GPU segmentation pass per case, including mandatory vertebral body segmentation followed by lumbar region-of-interest segmentation. Large abdominal CT volumes therefore benefit from GPUs with sufficient VRAM to reduce out-of-memory failures and avoid excessive reliance on low-memory workarounds. A 24 GB GPU is a practical minimum for routine use, while 32 GB or 48 GB provides more headroom for larger studies, method refinement, and future model experimentation.

The proposed workstation will support:

- Batch segmentation of CT datasets.
- Repeated re-processing during method refinement.
- Quality-control image generation.
- Statistical analysis in Python.
- Temporary staging of DICOM input and exported masks, analysis outputs, and QC images, with long-term retention on institutional storage rather than on the workstation.

## Recommended AI Workstation Specification

The recommended specification is a GPU-first AI workstation for long-term institutional research use. The purchase should prioritize GPU VRAM and throughput for 3D medical image segmentation because that is the main performance bottleneck for this project. Storage is intentionally minimal because the workstation is a processing node, not an imaging archive. DICOM studies are ingested for batch runs, and masks, analysis outputs, and QC images are exported back to institutional storage after processing.

### GPU Choice: RTX 6000 Ada Versus RTX 5090 And Alternatives

The GPU decision is not simply "buy the card with the most CUDA cores." The project needs reliable GPU memory for multi-pass CT segmentation and fast inference for batch research processing over multiple study cycles.

| GPU | VRAM | CUDA cores (approx.) | Strength for this project | Recommended role |
| --- | --- | --- | --- | --- |
| NVIDIA RTX 6000 Ada Generation | 48 GB GDDR6 ECC | 18,176 | Best single-GPU fit for large 3D CT volumes, dual TotalSegmentator passes, and long unattended batch jobs with ECC stability | Preferred for reliability and memory headroom |
| NVIDIA GeForce RTX 5090 | 32 GB GDDR7 | 21,760 | Fastest raw inference per dollar; strong for TotalSegmentator throughput | Strong alternative for speed-focused deployment |
| NVIDIA RTX 4090 / RTX 3090 | 24 GB | 16,384 / 10,496 | Minimum practical tier for GPU CT inference with careful memory management | Acceptable fallback |
| NVIDIA RTX A6000 | 48 GB | 10,752 | Older 48 GB workstation option; worth considering if institutionally available at favorable cost | Alternative 48 GB option |
| NVIDIA RTX PRO 6000 Blackwell | 96 GB GDDR7 ECC | 24,064 | Highest-capacity option, but typically a premium procurement path | Future upgrade path only |

Important clarification: among common single-GPU options for this project, the RTX 5090 currently has more CUDA cores and higher memory bandwidth than the RTX 6000 Ada. The RTX 6000 Ada is not faster because it is a "bigger" compute card. It is the better choice when 48 GB VRAM, ECC memory, lower power draw, and reduced risk of GPU out-of-memory failures on large CT studies matter more than peak consumer-GPU throughput.

The recommended procurement order is:

1. **First choice:** NVIDIA RTX 6000 Ada Generation for the best balance of memory headroom and long-running research stability.
2. **Second choice:** NVIDIA GeForce RTX 5090 when faster batch throughput is the priority and 32 GB VRAM is sufficient.
3. **Fallback:** NVIDIA RTX 4090 or RTX 3090 with 24 GB VRAM for minimum viable GPU processing.

### Target Configuration

| Component | Recommended Specification | Research Justification |
| --- | --- | --- |
| GPU | NVIDIA RTX 6000 Ada Generation 48 GB preferred; NVIDIA RTX 5090 32 GB as value/speed alternative; minimum RTX 4090 / RTX 3090 24 GB | 48 GB VRAM reduces out-of-memory risk on large abdominal CT volumes and supports multi-pass segmentation. RTX 5090 is faster for CT batch throughput but has less VRAM. |
| CUDA Support | Current NVIDIA driver with CUDA 12.x-compatible PyTorch stack | Required for GPU-accelerated PyTorch, TotalSegmentator, and nnU-Net workloads. |
| CPU | 8- to 16-core desktop or workstation CPU | Sufficient for DICOM conversion, preprocessing, and file operations. |
| System RAM | 64 GB DDR5 minimum; 96 GB preferred where available | Supports CT preprocessing, mask handling, statistics, and visualization during batch processing. |
| Local Storage | Single 1 TB to 2 TB NVMe Gen4 SSD | Holds OS, Python environment, segmentation model weights, and temporary processing space only. |
| Data Retention | Processing-in, processing-out workflow | Raw DICOM and long-term derived outputs remain on institutional storage. |
| Backup | Use existing institutional backup or network storage | Avoids duplicating long-term archive infrastructure on the workstation. |
| Network | 1 GbE minimum on motherboard | Adequate for staging active DICOM batches from institutional storage. |
| Operating System | Windows 11 Pro or Ubuntu LTS; WSL2 optional | Supports existing batch GUI, QC workflow, and reproducible AI tooling. |
| Power Supply | Quality PSU matched to selected GPU | RTX 6000 Ada has lower power demand than high-end consumer GPUs. |
| Cooling | Standard workstation chassis with adequate airflow | Supports long batch segmentation runs. |
| UPS | Basic UPS recommended | Protects active jobs during short power interruptions. |
| Security | TPM 2.0, full-disk encryption, administrator-controlled access | Supports responsible handling of sensitive temporary local processing. |
| Display | Use existing institutional monitor where possible | Avoids spending infrastructure funds on non-compute peripherals. |

### GPU Memory Requirements For 3D CT Segmentation

The GPU specification for this project should be interpreted as follows:

- Minimum acceptable configuration: 24 GB VRAM for routine GPU CT inference with careful memory settings and smaller active batch sizes.
- Strong configuration: 32 GB VRAM RTX 5090 for faster TotalSegmentator throughput and more reliable processing of typical abdominal CT studies.
- Preferred configuration: 48 GB RTX 6000 Ada for large-volume CT studies, repeated re-processing during pipeline refinement, and lower risk of GPU memory failures during dual-pass segmentation.
- Future upgrade path: RTX PRO 6000 Blackwell 96 GB or H100-class accelerators if later funded separately for expanded AI workloads.

### Minimum Acceptable Configuration

The minimum acceptable configuration should be:

- NVIDIA CUDA GPU with 24 GB VRAM.
- 12-core CPU.
- 64 GB system RAM.
- Single 1 TB NVMe SSD.
- 1 GbE networking.
- Basic UPS protection.

The minimum configuration should not drop below 24 GB GPU VRAM if reliable GPU CT batch processing remains a project requirement.

### Not Recommended

The following configurations are not recommended for this project:

- Consumer laptops, even with discrete GPUs, because thermal limits and low VRAM reduce reliability.
- GPUs with 8 GB to 16 GB VRAM, because they are likely to fail on larger CT volumes and dual-pass segmentation.
- CPU-only systems, because CT batch processing would be impractically slow.
- Large local HDD or NAS arrays in the base workstation purchase, because storage is not a project bottleneck.
- Premium non-essential peripherals or archive storage that do not improve segmentation performance.
- Assuming "more expensive GPU = more CUDA cores." RTX 6000 Ada costs more mainly because of 48 GB ECC VRAM, not because it beats RTX 5090 on core count.
- Long-term dependence on external vendor AI services when an equivalent local pipeline already exists and can be maintained in-house.

## Local Service Versus Vendor And Cloud Alternatives

The funding request should be understood as investment in durable institutional capability, not as a generic desktop purchase. The table below summarizes why a local AI workstation is a better fit for this project than common vendor or cloud alternatives.

| Approach | Typical advantages | Limitations for this project | Cost profile |
| --- | --- | --- | --- |
| Local AI workstation | Full workflow control, on-site data handling, repeatable batch processing, reusable across grants | Requires upfront procurement and technical maintenance | One-time infrastructure cost, low marginal cost per additional study |
| Commercial AI analysis vendor | Fast initial deployment, vendor-managed infrastructure | Black-box workflow, limited customization of L1 trabecular core logic, recurring license or per-study fees, data export constraints | Recurring OPEX increases with cohort size |
| Cloud GPU processing | Flexible compute scaling | Repeated upload/download of large DICOM datasets, governance review burden, ongoing hourly charges, less predictable runtime cost for iterative research | Pay-per-use; expensive for repeated reprocessing |
| Manual ROI or external measurement service | No hardware required | Slow, operator-dependent, difficult to scale across hundreds of CT studies | Staff time cost rises linearly with case volume |
| Hospital PACS-integrated AI module | Convenient clinical workflow integration | May not support research-specific outputs, custom QC tracking, or DEXA-matching data structures required by this study | Annual licensing and vendor dependency |

For an ongoing institutional research program, the local workstation model is more cost-effective because the marginal cost of processing an additional CT case approaches zero after infrastructure is in place. By contrast, vendor and cloud models often charge again for every re-run, every protocol experiment, and every quality-control cycle. That difference becomes important when the team must test segmentation settings, exclude failed cases, and regenerate analysis files multiple times before publication.

### Why Local Hosting Is The Better Long-Term Model

- **Research flexibility:** The team can modify segmentation settings, trabecular core erosion, export formats, and batch QC logic as the protocol evolves.
- **Governance:** Imaging data can be processed on-site and returned to approved institutional storage without routine off-site transfer.
- **Throughput:** Unattended overnight batch jobs are practical on dedicated local GPU hardware.
- **Sustainability:** The same workstation can support follow-on studies on sarcopenia, vertebral fracture risk, or other opportunistic CT biomarkers.
- **Financial predictability:** Infrastructure cost is known at procurement time, whereas vendor and cloud pricing can grow with study size and project duration.

## Cost-Effectiveness

The workstation is cost-effective because it is a shared research infrastructure asset rather than a single-use purchase. It will support the current osteoporosis screening project and future medical imaging studies without requiring a new external service contract each time the team launches a related project.

Compared with vendor-hosted analysis platforms, the local model avoids recurring per-study charges, subscription fees, and workflow restrictions that are poorly matched to academic research needs. Compared with cloud GPU services, local hosting avoids repeated data-transfer overhead, variable hourly billing, and governance delays associated with moving large DICOM datasets off-site. Compared with manual measurement services, automation reduces staff time and improves consistency across large cohorts.

The workstation also reduces personnel time spent waiting for slow CPU processing, troubleshooting memory failures, or working around GPUs with insufficient VRAM for 3D CT segmentation. Faster processing enables more cases to be included, improves statistical power, shortens time to analysis, and increases the likelihood that the project will produce publishable results within the funding period.

Over a multi-year horizon, the break-even point versus external services typically occurs once the institution would otherwise have paid repeatedly for batch processing, re-analysis, or licensed AI tools across multiple research waves. After that point, each additional validation or future patient cohort processed locally adds research value at low incremental cost.

## Data Governance And Security

Medical imaging research must be conducted under appropriate ethical and institutional approval. The workstation will be configured to support responsible data handling:

- Access restricted to authorized research personnel.
- Full-disk encryption enabled.
- Local firewall and endpoint protection enabled.
- No unnecessary external file-sharing services.
- Research datasets stored according to institutional policy on network or archive systems, not on the workstation long term.
- De-identification performed where required before analysis.
- Audit logs, processing logs, and quality-control outputs retained for reproducibility.
- Temporary working files deleted after export to approved institutional storage.

The system is intended to reduce unnecessary data retention on the workstation itself while still allowing controlled local GPU processing of medical imaging data.

## Project Timeline

| Phase | Duration | Activities | Milestones |
| --- | --- | --- | --- |
| Phase 1: Procurement And Setup | Months 1-2 | Procure workstation, install operating system, configure CUDA/PyTorch environment, install project dependencies, validate GPU processing. | Workstation operational; test CT case processed successfully. |
| Phase 2: Dataset Preparation | Months 2-4 | Identify eligible CT and DEXA cases, organize DICOM datasets, finalize inclusion criteria, prepare analysis registry. | Curated research dataset ready for batch processing. |
| Phase 3: Batch Segmentation | Months 4-7 | Run automated segmentation, generate masks, calculate HU statistics, create QC outputs. | Batch analysis and segmentation outputs generated. |
| Phase 4: Quality Control | Months 6-9 | Review segmentation overlays, mark success/failure, document failure modes, exclude invalid cases. | Clean analysis cohort finalized. |
| Phase 5: Statistical Analysis | Months 9-11 | Match CT metrics with DEXA, run correlation and classification analyses, evaluate thresholds. | Statistical results and figures generated. |
| Phase 6: Dissemination | Months 11-12 | Prepare abstract, manuscript, internal report, and future grant material. | Submission-ready research outputs completed. |

## Personnel And Roles

- Principal investigator: Oversees study design, ethics approval, clinical interpretation, and dissemination.
- Clinical imaging collaborator: Reviews CT suitability, segmentation quality, and clinical relevance of HU measurements.
- Research assistant or coordinator: Manages case lists, DEXA matching, documentation, and quality-control tracking.
- Technical lead or developer: Maintains the Python pipeline, workstation environment, batch processing, and reproducibility.
- Statistician or data analyst: Performs correlation, regression, ROC analysis, and manuscript-ready statistical reporting.

## Risk Management

| Risk | Mitigation Strategy |
| --- | --- |
| Segmentation failure in abnormal anatomy or limited field-of-view CT | Use visual QC workflow, document failure reasons, exclude invalid cases, and refine processing criteria. |
| Insufficient sample size after CT-DEXA matching | Expand date window if clinically justified, include additional CT protocols, or extend data collection period. |
| Hardware memory limits | Prioritize RTX 6000 Ada 48 GB where possible; otherwise RTX 5090 or 24 GB fallback; use low-memory CT processing modes and sequential batch scheduling where needed. |
| Data governance delays | Prepare protocol, de-identification plan, and storage plan early in Phase 1. |
| Software dependency changes | Freeze working Python environment, document versions, and maintain reproducible installation files. |
| Power interruption during batch jobs | Use UPS and resumable batch processing. |

## Deliverables

The funded project will deliver:

- A fully configured local AI workstation for medical image research.
- A documented software environment for CT segmentation and HU analysis.
- A curated CT-DEXA research dataset according to approved governance procedures.
- Batch-processed lumbar segmentation outputs and HU statistics.
- Quality-control records and failure-mode analysis.
- Statistical report on CT HU correlation with DEXA BMD.
- At least one conference abstract or manuscript draft.
- A reusable foundation for future opportunistic CT and medical imaging AI studies.

## Sustainability And Future Expansion

After completion of the initial validation study, the workstation can support additional projects that may extend benefit to patients:

- Prospective or operational studies testing opportunistic osteoporosis assessment in new patient cohorts.
- Fine-tuning segmentation models on local CT data.
- Evaluating multi-level lumbar HU measurements beyond L1.
- Developing automated CT-based sarcopenia or body composition workflows.
- Building interactive quality-control tools for radiology research and future clinical review workflows.
- Supporting student and trainee AI research projects.
- Creating reproducible pipelines for other DICOM-based imaging biomarkers.

The requested workstation will therefore strengthen institutional research capacity beyond the immediate project and preserve infrastructure for future patient-focused studies.

## Conclusion

This proposal requests funding for a dedicated local AI workstation to support an active opportunistic CT osteoporosis research program. The project addresses an important clinical problem: many patients at risk for osteoporosis are not screened before fracture, while routine CT images that could provide bone-health information already exist.

The research pipeline can automatically segment lumbar vertebrae, generate an L1 trabecular bone core, calculate HU statistics, and match outputs with DEXA BMD data. The workstation will first support retrospective validation and then remain available for future studies that may translate this capability into added benefit for patients, such as more systematic bone-health assessment from CT already performed during routine care. Hosting this workflow locally is more cost-effective and more scientifically appropriate than relying on recurring vendor or cloud services that charge per study, limit workflow customization, and increase data-governance complexity. The preferred GPU is the NVIDIA RTX 6000 Ada Generation because its 48 GB VRAM reduces out-of-memory risk on large 3D CT studies and supports reliable multi-pass segmentation. The RTX 5090 remains a strong speed-oriented alternative. Storage is limited to a fast local NVMe drive for temporary processing, while long-term data retention remains on institutional systems.

Funding this workstation will make the initial validation study feasible at meaningful scale, improve processing reliability, reduce analysis time, strengthen institutional research capacity, and create a reusable local AI platform for future medical imaging research with potential patient benefit.
