# Corpus sweep (rules only)

Generated 2026-11-03T21:11:29+00:00. Corpus: BioProBench (protocols.io + Protocol Exchange) (10518 records, 213 duplicate URLs and 156 empty protocols dropped).

## Funnel

| stage | count | share |
|---|---|---|
| protocols | 10149 | |
| steps | 246169 | 100% |
| grounded by rules | 48600 | 19.7% |
| needs LLM | 122174 | 49.6% |
| out of scope (rules) | 69348 | 28.2% |
| planned into ops | 29436 | 12.0% |
| planned liquid steps | 19031 | 7.7% |

| protocols | count |
|---|---|
| with at least one planned step | 5518 |
| task candidates (judged liquid step + buildable scene) | 2003 |
| with a task segment (3+ consecutive executable steps, one liquid) | 981 |
| fully planned (every step executable) | 7 |

Scene builder on protocols with an executable step:

| result | protocols |
|---|---|
| ok | 3131 |
| missing_asset: tube_vial needs a tube rack; pipetting families have none | 1284 |
| over_capacity: no free trough slot left in the culture scene | 749 |
| family_conflict: program needs both culture dishes and a readable plate | 201 |
| over_capacity: no free plate slot left in the culture scene | 66 |
| over_capacity: no free trough slot left in the protein scene | 30 |
| unreachable: centrifuge_tube_15ml rim at z=0.120 m: a held pipette puts t | 15 |
| missing_asset: balance cannot hold liquid in this scene | 12 |
| over_capacity: 500 uL exceeds the protein pipette (200 uL) | 8 |
| over_capacity: 1000 uL exceeds the protein pipette (200 uL) | 4 |
| over_capacity: 250 uL exceeds the protein pipette (200 uL) | 3 |
| over_capacity: 300 uL exceeds the protein pipette (200 uL) | 3 |

Longest run of consecutive planned steps per protocol:

| run | protocols |
|---|---|
| 0 | 4631 |
| 1 | 2926 |
| 2 | 1561 |
| 3 | 600 |
| 4 | 250 |
| 5 | 114 |
| 6 | 37 |
| 7 | 19 |
| 8 | 5 |
| 9 | 2 |
| 10+ | 4 |

## What blocks steps

`<skill>:unbound` means the skill exists but the step gives no binding (usually no volume); `unclustered` is mostly text that is not a robot action (notes, design, analysis).

| capability | protocols | steps |
|---|---|---|
| unclustered | 9230 | 85449 |
| centrifuge | 4293 | 20360 |
| transfer:unbound | 4001 | 9798 |
| pour | 2688 | 6334 |
| wash | 2682 | 8211 |
| cool | 2328 | 4792 |
| resuspend | 2233 | 5199 |
| vortex | 2141 | 5446 |
| mix | 1854 | 3677 |
| image | 1830 | 5332 |
| dilute | 1743 | 2757 |
| lyse | 1711 | 3874 |
| thermocycle | 1558 | 4248 |
| dry | 1453 | 2582 |
| incubate:unbound | 1313 | 2255 |
| stain | 1262 | 2622 |
| heat | 1251 | 1938 |
| dilute:unbound | 1202 | 1782 |
| cold_store | 1144 | 1662 |
| fix | 1118 | 1878 |
| freeze | 1071 | 1622 |
| inspect:unbound | 1043 | 1628 |
| passage | 1026 | 1910 |
| discard | 991 | 2051 |
| move_to:unbound | 946 | 1792 |
| weigh | 929 | 1548 |
| dissolve | 873 | 1333 |
| shake | 872 | 1612 |
| aliquot | 782 | 1171 |
| rinse | 693 | 1432 |

## Unlock order

Adding capabilities greedily: each row is the one that gives the most further protocols a task segment (3+ consecutive executable steps, one liquid). Scenes are not re-checked here.

Protocols with a segment today: 981.

| add | unlocks | protocols with a segment after |
|---|---|---|
| centrifuge | 854 | 1835 |
| transfer:unbound | 646 | 2481 |
| wash | 382 | 2863 |
| vortex | 243 | 3106 |
| pour | 224 | 3330 |
| dilute | 218 | 3548 |
| cool | 200 | 3748 |
| resuspend | 188 | 3936 |
| dilute:unbound | 144 | 4080 |
| mix | 140 | 4220 |
| thermocycle | 125 | 4345 |
| image | 124 | 4469 |
| lyse | 111 | 4580 |
| aspirate:unbound | 99 | 4679 |
| stain | 76 | 4755 |

## LLM pass sizing

- needs_llm steps in total: 122174 (about $1367.19 at placeholder prices)
- near-complete protocols (every other step planned, at most 3 needs_llm steps): 24, 46 calls (about $0.51)

## Frozen tasks

37 of 40 top candidates froze into validated tasks (oracle passes, two fresh replays agree, doing nothing fails).

| not frozen because | count |
|---|---|
| trivial | 2 |
| oracle_fails_checks | 1 |

## Top candidates

| protocol | title | steps | planned | liquid | longest run |
|---|---|---|---|---|---|
| [Protocol.io-7583](https://www.protocols.io/view/a-sequencing-and-subtyping-protocol-for-influenza-crv3v68n) | A sequencing and subtyping protocol for Influenza A and B viruses usin | 47 | 9 | 9 | 2 |
| [Protocol.io-7385](https://www.protocols.io/view/ldh-cytotoxicity-assay-dukn6uve) | LDH cytotoxicity assay | 12 | 7 | 6 | 5 |
| [Protocol.io-5721](https://www.protocols.io/view/upitt-tristate-sennet-tmc-total-rna-isolation-amp-c2yzyfx6) | UPitt TriState SenNet TMC Total RNA isolation &amp; quality analysis f | 53 | 6 | 6 | 4 |
| [Protocol.io-8080](https://www.protocols.io/view/u-mass-non-esterified-fatty-acids-x3hfqj6) | U Mass - Non-esterified fatty acids | 15 | 6 | 6 | 4 |
| [Protocol.io-10100](https://www.protocols.io/view/jax-dpc-genotyping-and-selection-of-whole-gene-kno-dyt97wr6) | JAX_DPC: Genotyping and selection of whole-gene knockout (KO) and crit | 36 | 6 | 6 | 3 |
| [Protocol.io-265](https://www.protocols.io/view/u-michigan-retinal-cell-death-yaafsae) | U Michigan - Retinal Cell Death | 17 | 7 | 6 | 3 |
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
| [Protocol.io-6729](https://www.protocols.io/view/2-step-pcr-mixture-and-conditions-barcoded-head-pr-cvkcw4sw) | 2-step PCR mixture and conditions (Barcoded-head primers for seqs pool | 17 | 5 | 5 | 3 |
