# Corpus sweep (rules only)

Generated 2026-11-04T23:26:00+00:00. Corpus: BioProBench (protocols.io + Protocol Exchange) (10518 records, 213 duplicate URLs and 156 empty protocols dropped).

## Funnel

| stage | count | share |
|---|---|---|
| protocols | 10149 | |
| steps | 246169 | 100% |
| grounded by rules | 67133 | 27.3% |
| needs LLM | 117350 | 47.7% |
| out of scope (rules) | 55639 | 22.6% |
| planned into ops | 31177 | 12.7% |
| planned liquid steps | 20594 | 8.4% |

| protocols | count |
|---|---|
| with at least one planned step | 5595 |
| task candidates (judged liquid step + buildable scene) | 1949 |
| with a task segment (3+ consecutive executable steps, one liquid) | 1130 |
| fully planned (every step executable) | 7 |

Scene builder on protocols with an executable step:

| result | protocols |
|---|---|
| ok | 3057 |
| missing_asset: tube_vial needs a tube rack; pipetting families have none | 1386 |
| over_capacity: no free trough slot left in the culture scene | 775 |
| family_conflict: program needs both culture dishes and a readable plate | 202 |
| over_capacity: no free plate slot left in the culture scene | 89 |
| over_capacity: no free trough slot left in the protein scene | 28 |
| unreachable: centrifuge_tube_15ml rim at z=0.120 m: a held pipette puts t | 17 |
| missing_asset: balance cannot hold liquid in this scene | 11 |
| over_capacity: 500 uL exceeds the protein pipette (200 uL) | 8 |
| over_capacity: 1000 uL exceeds the protein pipette (200 uL) | 4 |
| over_capacity: 250 uL exceeds the protein pipette (200 uL) | 3 |
| over_capacity: 300 uL exceeds the protein pipette (200 uL) | 3 |

Longest run of consecutive planned steps per protocol:

| run | protocols |
|---|---|
| 0 | 4554 |
| 1 | 2834 |
| 2 | 1581 |
| 3 | 689 |
| 4 | 279 |
| 5 | 130 |
| 6 | 42 |
| 7 | 27 |
| 8 | 7 |
| 9 | 2 |
| 10+ | 4 |

## What blocks steps

`<skill>:unbound` means the skill exists but the step gives no binding (usually no volume); `unclustered` is mostly text that is not a robot action (notes, design, analysis).

| capability | protocols | steps |
|---|---|---|
| unclustered | 9230 | 85449 |
| centrifuge | 4191 | 18623 |
| transfer:unbound | 4105 | 10271 |
| pour | 2688 | 6334 |
| wash | 2682 | 8211 |
| cool | 2328 | 4792 |
| resuspend | 2209 | 5129 |
| vortex | 2089 | 5168 |
| mix | 1854 | 3677 |
| image | 1828 | 5328 |
| dilute | 1720 | 2692 |
| lyse | 1679 | 3779 |
| thermocycle | 1558 | 4248 |
| dry | 1453 | 2582 |
| incubate:unbound | 1334 | 2292 |
| stain | 1256 | 2600 |
| heat | 1251 | 1938 |
| dilute:unbound | 1217 | 1809 |
| cold_store | 1144 | 1662 |
| fix | 1114 | 1864 |
| freeze | 1071 | 1622 |
| inspect:unbound | 1055 | 1645 |
| passage | 1021 | 1897 |
| discard | 991 | 2051 |
| move_to:unbound | 961 | 1817 |
| weigh | 927 | 1545 |
| shake | 872 | 1612 |
| dissolve | 872 | 1331 |
| aliquot | 782 | 1171 |
| rinse | 693 | 1432 |

## Unlock order

Adding capabilities greedily: each row is the one that gives the most further protocols a task segment (3+ consecutive executable steps, one liquid). Scenes are not re-checked here.

Protocols with a segment today: 1130.

| add | unlocks | protocols with a segment after |
|---|---|---|
| centrifuge | 801 | 1931 |
| transfer:unbound | 644 | 2575 |
| wash | 376 | 2951 |
| vortex | 240 | 3191 |
| pour | 220 | 3411 |
| cool | 209 | 3620 |
| dilute | 201 | 3821 |
| resuspend | 185 | 4006 |
| dilute:unbound | 144 | 4150 |
| mix | 139 | 4289 |
| thermocycle | 125 | 4414 |
| image | 125 | 4539 |
| lyse | 108 | 4647 |
| aspirate:unbound | 98 | 4745 |
| stain | 79 | 4824 |

## Task families

Steps planned for each task scene, and how many its scene builder accepts.

| family | ops | in protocols | accepted | in protocols | rejected because |
|---|---|---|---|---|---|
| weigh | 307 | 187 | 2 | 2 | over_capacity 266, missing_asset 39 |
| centrifuge | 15519 | 3894 | 10466 | 3223 | missing_asset 5042, over_capacity 11 |
| prep_spin | 2054 | 1072 | 611 | 422 | missing_asset 879, ambiguous 436, over_capacity 108, out_of_range 15, unchecked 5 |

Accepted weigh candidates (first 15):

| protocol | title | step | parameters |
|---|---|---|---|
| [Protocol.io-2332](https://www.protocols.io/view/protocol-for-safe-affordable-and-reproducible-isol-bxavpie6) | Protocol for Safe, Affordable, and Reproducible Isolati | 5 | mass_g=0.9, reagent=sodium chloride (NaCl) |
| [Protocol.io-5260](https://www.protocols.io/view/protocol-for-safe-affordable-and-reproducible-isol-chjet4je) | Protocol for Safe, Affordable, and Reproducible Isolati | 5 | mass_g=0.9, reagent=sodium chloride (NaCl) |

Accepted centrifuge candidates (first 15):

| protocol | title | step | parameters |
|---|---|---|---|
| [Protocol-exchange-3](https://doi.org/10.21203/rs.3.pex-1234/v2) | Small-Scale Determination of Total Phenols, Tannins, an | 11 | tubes=2, time_s=240.0, speed=5000 rpm |
| [Protocol-exchange-3](https://doi.org/10.21203/rs.3.pex-1234/v2) | Small-Scale Determination of Total Phenols, Tannins, an | 16 | tubes=2, time_s=600.0, speed=5000 rpm |
| [Protocol-exchange-4](https://doi.org/10.21203/rs.3.pex-2074/v1) | Isolation of DNA by EdU-labelling for Mass Spectrometry | 15 | tubes=2, time_s=120.0 |
| [Protocol-exchange-4](https://doi.org/10.21203/rs.3.pex-2074/v1) | Isolation of DNA by EdU-labelling for Mass Spectrometry | 18 | tubes=2, time_s=120.0 |
| [Protocol-exchange-4](https://doi.org/10.21203/rs.3.pex-2074/v1) | Isolation of DNA by EdU-labelling for Mass Spectrometry | 50 | tubes=2, time_s=30.0 |
| [Protocol-exchange-4](https://doi.org/10.21203/rs.3.pex-2074/v1) | Isolation of DNA by EdU-labelling for Mass Spectrometry | 76 | tubes=2, time_s=600.0 |
| [Protocol-exchange-4](https://doi.org/10.21203/rs.3.pex-2074/v1) | Isolation of DNA by EdU-labelling for Mass Spectrometry | 82 | tubes=2 |
| [Protocol-exchange-5](https://doi.org/10.21203/rs.3.pex-1440/v1) | Generation of cell lines for the exon-specific isoform  | 26 | tubes=2 |
| [Protocol-exchange-5](https://doi.org/10.21203/rs.3.pex-1440/v1) | Generation of cell lines for the exon-specific isoform  | 52 | tubes=2 |
| [Protocol-exchange-6](https://doi.org/10.21203/rs.3.pex-1936/v2) | Preparing Active Recombinant AEP free of Acidactivation | 14 | tubes=2, time_s=900.0, speed=500 rpm |
| [Protocol-exchange-6](https://doi.org/10.21203/rs.3.pex-1936/v2) | Preparing Active Recombinant AEP free of Acidactivation | 15 | tubes=2 |
| [Protocol-exchange-7](https://doi.org/10.21203/rs.3.pex-1354/v1) | Automated, quantitative Fast FLIM-FRET (qF3): A step-by | 7 | tubes=2, time_s=900.0, speed=4500xg |
| [Protocol-exchange-7](https://doi.org/10.21203/rs.3.pex-1354/v1) | Automated, quantitative Fast FLIM-FRET (qF3): A step-by | 12 | tubes=2 |
| [Protocol-exchange-8](https://doi.org/10.21203/rs.3.pex-1659/v1) | An optimized protocol for propidium monoazide treatment | 16 | tubes=2, time_s=780.0, speed=5,000×g |
| [Protocol-exchange-10](https://doi.org/10.21203/rs.3.pex-1695/v1) | Antibody-guided Chromatin Tagmentation for Two or More  | 9 | tubes=2, time_s=120.0, speed=850 g |

Accepted prep_spin candidates (first 15):

| protocol | title | step | parameters |
|---|---|---|---|
| [Protocol-exchange-43](https://doi.org/10.21203/rs.2.14634/v1) | Time-Resolved Serial Femtosecond Crystallography at the | 18 | additions=1, reagents=enterokinase |
| [Protocol-exchange-144](https://doi.org/10.1038/protex.2015.067) | An improved method for extraction and quantification of | 20 | additions=1, reagents=chloroform:isoamyl alcohol solution |
| [Protocol-exchange-170](https://doi.org/10.1038/protex.2013.074) | Fluorescent DNA probes for ultra-sensitive strandspecif | 14 | additions=1, reagents=NaOH |
| [Protocol-exchange-226](https://doi.org/10.1038/protex.2011.209) | Preparation of materials for flexizyme reactions and ge | 18 | additions=1, reagents=ethanol |
| [Protocol-exchange-226](https://doi.org/10.1038/protex.2011.209) | Preparation of materials for flexizyme reactions and ge | 41 | additions=1, reagents=ethanol |
| [Protocol-exchange-226](https://doi.org/10.1038/protex.2011.209) | Preparation of materials for flexizyme reactions and ge | 80 | additions=1, reagents=ethanol |
| [Protocol-exchange-226](https://doi.org/10.1038/protex.2011.209) | Preparation of materials for flexizyme reactions and ge | 102 | additions=1, reagents=ethanol |
| [Protocol-exchange-228](https://doi.org/10.1038/protex.2011.216) | Characterisation and quantitation expression analysis o | 28 | additions=1, reagents=hydrolysation solution |
| [Protocol-exchange-237](https://doi.org/10.1038/nprot.2010.151) | A complete protocol for genome-scale analysis of replic | 39 | additions=1, reagents=ethanol |
| [Protocol-exchange-461](https://doi.org/10.21203/rs.3.pex-1246/v1) | Mouse fetal liver erythroid population separation using | 39 | additions=1, reagents=ice-cold staining solution |
| [Protocol-exchange-485](https://doi.org/10.21203/rs.3.pex-896/v1) | Direct Synthesis of Gold Nanoparticles on Cysteinerich  | 30 | additions=1, reagents=PBS-A |
| [Protocol-exchange-554](https://doi.org/10.1038/protex.2019.010) | Protocol for CRISPR/Cas9-based knock-in using the VIKIN | 21 | additions=1, reagents=Tris HCl |
| [Protocol-exchange-577](https://doi.org/10.1038/protex.2018.066) | Multiplexed Imaging of Neuronal Synapses Using Nucleic  | 37 | additions=2, reagents=sodium acetate+isopropanol |
| [Protocol-exchange-585](https://doi.org/10.1038/protex.2018.059) | High Molecular Weight DNA Extraction from Recalcitrant  | 20 | additions=1, reagents=cold NIB |
| [Protocol-exchange-589](https://doi.org/10.1038/protex.2018.026) | Hox/Wnt interact to pattern the primary body axis of an | 21 | additions=1, reagents=RNase-free 70% EtOH |

## LLM pass sizing

- needs_llm steps in total: 117350 (about $1313.2 at placeholder prices)
- near-complete protocols (every other step planned, at most 3 needs_llm steps): 26, 47 calls (about $0.53)

## Top candidates

| protocol | title | steps | planned | liquid | longest run |
|---|---|---|---|---|---|
| [Protocol.io-7385](https://www.protocols.io/view/ldh-cytotoxicity-assay-dukn6uve) | LDH cytotoxicity assay | 12 | 7 | 6 | 5 |
| [Protocol.io-8080](https://www.protocols.io/view/u-mass-non-esterified-fatty-acids-x3hfqj6) | U Mass - Non-esterified fatty acids | 15 | 6 | 6 | 4 |
| [Protocol.io-8094](https://www.protocols.io/view/uc-davis-catalase-ygbftsn) | UC Davis - Catalase | 10 | 7 | 6 | 4 |
| [Protocol.io-10100](https://www.protocols.io/view/jax-dpc-genotyping-and-selection-of-whole-gene-kno-dyt97wr6) | JAX_DPC: Genotyping and selection of whole-gene knockout (KO) and crit | 36 | 6 | 6 | 3 |
| [Protocol.io-2123](https://www.protocols.io/view/reporter-titrations-for-transfection-of-micromonas-8p8hvrw) | Reporter titrations for transfection of Micromonas commoda CCMP2709 | 41 | 6 | 6 | 3 |
| [Protocol.io-265](https://www.protocols.io/view/u-michigan-retinal-cell-death-yaafsae) | U Michigan - Retinal Cell Death | 17 | 7 | 6 | 3 |
| [Protocol.io-4983](https://www.protocols.io/view/single-gene-short-term-crispr-ko-viability-assay-bc6jizcn) | Single-gene short-term CRISPR ko viability assay | 57 | 6 | 6 | 3 |
| [Protocol.io-6868](https://www.protocols.io/view/scd40l-elisa-assay-cysqxwdw) | sCD40L ELISA Assay | 15 | 7 | 6 | 3 |
| [Protocol.io-585](https://www.protocols.io/view/elisa-for-assessing-the-burden-of-neutrophil-extra-bmp7k5rn) | ELISA for assessing the burden of Neutrophil Extracellular Traps (NETs | 15 | 6 | 6 | 2 |
| [Protocol.io-6234](https://www.protocols.io/view/assay-protocol-384-well-plates-dyri7v4e) | Assay Protocol (384-well plates) | 18 | 6 | 6 | 2 |
| [Protocol.io-6243](https://www.protocols.io/view/assay-protocol-96-well-plates-dyrh7v36) | Assay Protocol (96-well plates) | 19 | 6 | 6 | 2 |
| [Protocol.io-7506](https://www.protocols.io/view/virus-igm-antibody-detection-bdasi2ee) | Virus IgM antibody detection | 34 | 6 | 6 | 2 |
| [Protocol.io-965](https://www.protocols.io/view/highly-parallel-droplet-sample-preparation-for-sin-b8uyrwxw) | Highly Parallel Droplet Sample Preparation for Single Cell Proteomics | 50 | 6 | 6 | 2 |
| [Protocol.io-2805](https://www.protocols.io/view/Sytoxgreen-staining-DNA-stain-for-flow-cytometry-fepbjdn) | Sytoxgreen staining (DNA stain, for flow cytometry) | 8 | 5 | 5 | 5 |
| [Protocol.io-7183](https://www.protocols.io/view/preparation-and-imaging-of-lipid-bilayer-coated-si-ddca22se) | Preparation and imaging of lipid bilayer-coated silica microspheres | 29 | 5 | 5 | 5 |
| [Protocol-exchange-823](https://doi.org/10.1038/protex.2013.009) | Linear Amplification Mediated PCR for the retrieval of lentiviral vect | 30 | 5 | 5 | 4 |
| [Protocol.io-2185](https://www.protocols.io/view/rtpcr-amplification-of-sars-cov2-whole-genome-for-bew8jfhw) | RTPCR Amplification of SARS-CoV2 Whole Genome for Illumina NGS | 20 | 5 | 5 | 4 |
| [Protocol.io-5400](https://www.protocols.io/view/diagnosis-of-taenia-solium-infections-based-on-mai-bzx6p7re) | Diagnosis of Taenia solium infections based on “mail order” RNA-sequen | 18 | 5 | 5 | 4 |
| [Protocol.io-5413](https://www.protocols.io/view/automated-protein-quantification-with-the-biomek-f-b3grqjv6) | Automated Protein Quantification with the Biomek-FX Liquid Handler Sys | 16 | 5 | 5 | 4 |
| [Protocol.io-1229](https://www.protocols.io/view/expansion-and-maintenance-of-human-induced-pluripo-cpvgvn3w) | Expansion and maintenance of human induced pluripotent stem cells (iPS | 25 | 6 | 5 | 3 |
| [Protocol.io-2900](https://www.protocols.io/view/dna-quantification-with-picogreen-kqtcvwn) | DNA quantification with PicoGreen | 10 | 5 | 5 | 3 |
| [Protocol.io-2957](https://www.protocols.io/view/dna-quantification-with-picogreen-q2wdyfe) | DNA quantification with PicoGreen | 10 | 5 | 5 | 3 |
| [Protocol.io-5981](https://www.protocols.io/view/lminex-milliplex-soluble-cytokine-receptor-13-plex-hvib64e) | Lminex Milliplex Soluble Cytokine Receptor 13-plex | 31 | 5 | 5 | 3 |
| [Protocol.io-6254](https://www.protocols.io/view/isolation-of-classical-monocytes-from-pbmcs-mb-kit-dzew73fe) | Isolation of Classical Monocytes from PBMCs (MB Kit No. 130-117-337) | 27 | 5 | 5 | 3 |
| [Protocol.io-6570](https://www.protocols.io/view/2-step-pcr-mixture-and-conditions-barcoded-head-pr-crpbv5in) | 2-step PCR mixture and conditions (Barcoded-head primers for seqs pool | 18 | 5 | 5 | 3 |
