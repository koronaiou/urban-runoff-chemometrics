"""
compound_annotation.py
======================
Expert-curated annotation of the 175 compounds identified (confidence levels 1,
2a and 2b sensu Schymanski et al. 2014) in Thessaloniki urban runoff.

Each compound is mapped to:
  * `source_class`  - the emission source / use category it diagnoses
  * `priority`      - True if the compound is a recognised emerging contaminant
                      of regulatory or ecotoxicological concern

The annotation is manual and based on the reported use of each substance; it is
NOT derived from the data and is therefore independent of any model result.
Users should treat it as a working hypothesis and adapt it to their own
source-apportionment framework.

Source classes
--------------
TRWP          Tire & road wear particles, rubber vulcanisation chemicals
PFAS          Per- and polyfluoroalkyl substances
OPE           Organophosphate esters (flame retardants / plasticisers)
PLASTIC       Plastic additives, phthalates, slip agents, bisphenols
SURFACTANT    Surfactants, their transformation products, PEG/PPG oligomers
CORROSION     Corrosion inhibitors (benzotriazoles)
PPCP          Pharmaceuticals and personal-care products
LIFESTYLE     Tobacco, caffeine, illicit drugs, human excretion markers
PESTICIDE     Pesticides, biocides and their transformation products
INDUSTRIAL    Other industrial chemicals, solvents, dye intermediates
NATURAL       Biogenic / plant / food-derived compounds (non-contaminant)
UNKNOWN       Tentative identifications without an assigned use
"""

CLASS_ORDER = [
    "TRWP", "PFAS", "OPE", "PLASTIC", "SURFACTANT", "CORROSION",
    "PPCP", "LIFESTYLE", "PESTICIDE", "INDUSTRIAL", "NATURAL", "UNKNOWN",
]

CLASS_LABEL = {
    "TRWP": "Tire & road wear / rubber",
    "PFAS": "PFAS",
    "OPE": "Organophosphate esters",
    "PLASTIC": "Plastic additives & phthalates",
    "SURFACTANT": "Surfactants & TPs",
    "CORROSION": "Corrosion inhibitors",
    "PPCP": "Pharmaceuticals & PCPs",
    "LIFESTYLE": "Lifestyle / wastewater markers",
    "PESTICIDE": "Pesticides & biocides",
    "INDUSTRIAL": "Other industrial chemicals",
    "NATURAL": "Natural / biogenic",
    "UNKNOWN": "Unassigned",
}

# compound name -> (source_class, priority_contaminant, short note)
ANNOTATION = {
    "(-)-3-Hydroxydodecanoic acid": ("NATURAL", False, "microbial/biogenic hydroxy fatty acid"),
    "(1R,4aS,5R,8aS)-5-(5-hydroxy-3-methylpentyl)-1,4a-dimethyl-6-methylidene-decahydronaphthalene-1-carboxylic acid": ("NATURAL", False, "labdane-type plant diterpenoid"),
    "1S,4S,5R,10S,13S,14R,19S,20R)-10-hydroxy-4,5,9,9,13,19,20-heptamethyl-24-oxahexacyclo[15,5,2,01,18,04,17,05,14,08,13]tetracos-15-en-23-one": ("NATURAL", False, "plant triterpenoid"),
    "(2S)-2-Nonanyl hydrogen sulfate": ("SURFACTANT", False, "secondary alkyl sulfate surfactant"),
    "(2Z)-2-Octyl-2-pentenedioic acid": ("INDUSTRIAL", False, "alkenyl succinic acid, paper/lubricant additive"),
    "(4-Isopropenyl-1-cyclohexen-1-yl)methyl formate": ("PPCP", False, "perillyl formate, fragrance ingredient"),
    "(R,S)-Anatabine": ("LIFESTYLE", False, "minor tobacco alkaloid"),
    "(S)-Nicotine": ("LIFESTYLE", True, "tobacco marker"),
    "1,4a-dimethyl-9-oxo-7-(propan-2-yl)-1,2,3,4,4a,9,10,10a-octahydrophenanthrene-1-carboxylic acid": ("NATURAL", False, "7-oxo-dehydroabietic acid, conifer resin / rosin"),
    "12-Oxo phytodienoic acid": ("NATURAL", False, "plant jasmonate precursor"),
    "1-Hydroperfluoroheptane": ("PFAS", True, "polyfluorinated alkane"),
    "1-Linoleoyl glycerol": ("NATURAL", False, "monoacylglycerol"),
    "1-Tetradecylamine": ("SURFACTANT", False, "fatty amine, cationic surfactant"),
    "2,4-dimethyl-2,3-dihydrochromeno[4,3-c]pyrazol-3-one": ("INDUSTRIAL", False, "synthetic heterocycle"),
    "20-Hydroxy-(5Z,8Z,11Z,14Z)-eicosatetraenoic acid": ("NATURAL", False, "eicosanoid"),
    "2-Aminobenzothiazole": ("TRWP", True, "benzothiazole, rubber vulcanisation TP"),
    "6PPD-q": ("TRWP", True, "6PPD-quinone, acutely toxic tire antiozonant TP"),
    "2-Benzothiazolsulfonic acid": ("TRWP", True, "benzothiazole oxidation TP"),
    "2-Butyloctyl dihydrogen phosphate": ("OPE", False, "branched alkyl phosphate"),
    "2-Hexyldecyl dihydrogen phosphate": ("OPE", False, "branched alkyl phosphate"),
    "2-Hydroxyacetophenone sulfate": ("INDUSTRIAL", False, "sulfated aromatic ketone"),
    "2-Hydroxybenzothiazole": ("TRWP", True, "benzothiazole, rubber/road runoff marker"),
    "2-Naphthalenesulfonic acid": ("INDUSTRIAL", False, "dye intermediate / concrete superplasticiser"),
    "3-(16-Methylheptadecyl)dihydro-2,5-furandione": ("INDUSTRIAL", False, "alkenyl succinic anhydride"),
    "3,5-Dinitro-o-cresol (DNOC)": ("PESTICIDE", True, "banned dinitrophenol herbicide/insecticide"),
    "3,5-Dioxo-4-oxatricyclo[5,2,2,0~2,6~]undec-10-ene-8-carboxylic acid": ("INDUSTRIAL", False, "polycyclic anhydride"),
    "3,5-di-tert-Butyl-4-hydroxybenzaldehyde": ("INDUSTRIAL", False, "BHT antioxidant transformation product"),
    "3-Methoxyphenylacetic acid": ("NATURAL", False, "plant/human metabolite"),
    "3-methyl-5-phenylpyridazine": ("INDUSTRIAL", False, "synthetic heterocycle"),
    "3-Phenoxybenzoic acid ": ("PESTICIDE", True, "common pyrethroid transformation product"),
    "3-Phenyl-1-(2,4,6-trihydroxyphenyl)-1-propanone": ("NATURAL", False, "phloretin-type dihydrochalcone"),
    "3-Phenyl-2,3-dihydro-1,3-benzothiazol-2-imine": ("TRWP", True, "benzothiazole derivative, rubber"),
    "4,4′-di-tert-butyl-2,2′bipyridyl": ("INDUSTRIAL", False, "metal-complex ligand"),
    "4,4-Bipyridine ": ("INDUSTRIAL", False, "industrial intermediate"),
    "Bisphenol S": ("PLASTIC", True, "bisphenol A substitute, endocrine activity"),
    "4-Dodecylbenzenesulfonic acid": ("SURFACTANT", True, "linear alkylbenzene sulfonate (LAS)"),
    "4-Ethynylaniline": ("INDUSTRIAL", False, "synthesis intermediate"),
    "4-Indolecarbaldehyde": ("NATURAL", False, "indole metabolite"),
    "4-Methylbenzotriazole": ("CORROSION", True, "tolyltriazole, de-icer & dishwasher additive"),
    "4-Nitrosodiphenylamine": ("TRWP", True, "rubber antiozonant transformation product"),
    "5-(6-hydroxy-6-methyloctyl)-2,5-dihydrofuran-2-one": ("NATURAL", False, "butenolide"),
    "5,6-Dimethylbenzimidazole": ("NATURAL", False, "vitamin B12 moiety"),
    "Tectorigenin": ("NATURAL", False, "plant isoflavone"),
    "6-Amino-7-hydroxynaphthalene-2-sulfonic acid": ("INDUSTRIAL", False, "J-acid, azo-dye intermediate"),
    "6-Hydroxy-1-naphthalenesulfonic acid": ("INDUSTRIAL", False, "dye intermediate"),
    "7-OXOCHOLESTEROL": ("NATURAL", False, "cholesterol oxidation product"),
    "8-(4-Sulfophenyl) octanoic acid": ("SURFACTANT", True, "sulfophenyl carboxylate, LAS degradation TP"),
    "8,8-dimethyl-2H,8H-pyrano[3,2-g]chromen-2-one": ("NATURAL", False, "plant coumarin"),
    "8H-Perfluorooctane": ("PFAS", True, "polyfluorinated alkane"),
    "Abietic acid": ("NATURAL", False, "conifer resin acid"),
    "Acetylsalicylic acid": ("PPCP", True, "analgesic"),
    "Antiarol": ("NATURAL", False, "plant phenol"),
    "Benzothiazole": ("TRWP", True, "vulcanisation accelerator / road runoff marker"),
    "Benzotriazole": ("CORROSION", True, "corrosion inhibitor, ubiquitous urban contaminant"),
    "Benzoylecgonine": ("LIFESTYLE", True, "cocaine metabolite, wastewater marker"),
    "Bis(2-ethylhexyl) amine": ("INDUSTRIAL", False, "industrial amine"),
    "Bis(2-ethylhexyl) terephthalate": ("PLASTIC", True, "DEHT, DEHP substitute plasticiser"),
    "Bis(3,5,5-trimethylhexyl) phthalate": ("PLASTIC", True, "branched phthalate plasticiser"),
    "Caffeine": ("LIFESTYLE", True, "classic domestic wastewater tracer"),
    "Cannabidiolic acid": ("LIFESTYLE", False, "cannabis marker"),
    "Cetrimonium": ("SURFACTANT", True, "CTAB, quaternary ammonium biocide"),
    "Citroflex 4": ("PLASTIC", False, "tributyl citrate plasticiser"),
    "Citroflex A-4": ("PLASTIC", False, "acetyl tributyl citrate plasticiser"),
    "Cocaine": ("LIFESTYLE", True, "illicit drug, wastewater marker"),
    "Columbianetin": ("NATURAL", False, "plant coumarin"),
    "Cotinine": ("LIFESTYLE", True, "nicotine metabolite"),
    "cyclohexylformamide": ("INDUSTRIAL", False, "industrial amide"),
    "DEET": ("PPCP", True, "insect repellent, urban runoff tracer"),
    "Deoxycholic Acid": ("LIFESTYLE", True, "faecal bile acid marker"),
    "Dibenzylamine": ("TRWP", False, "vulcanisation accelerator degradation product"),
    "Dicyclohexylamine": ("TRWP", False, "rubber accelerator / corrosion inhibitor"),
    "Didecyldimethylammonium": ("SURFACTANT", True, "DDAC, quaternary ammonium biocide"),
    "Diisobutyl Phosphate": ("OPE", True, "TiBP transformation product"),
    "Dinoterb": ("PESTICIDE", True, "banned dinitrophenol herbicide"),
    "Diphenyl sulfone": ("INDUSTRIAL", False, "polymer intermediate"),
    "Diphenylamine": ("TRWP", True, "rubber antioxidant, also post-harvest fungicide"),
    "Docosahexaenoic acid ethyl ester": ("NATURAL", False, "food lipid"),
    "Dodecyl sulfate": ("SURFACTANT", True, "SDS, anionic surfactant"),
    "e-Decalactone": ("PPCP", False, "fragrance / flavour lactone"),
    "Eicosatetraynoic acid": ("NATURAL", False, "fatty acid analogue"),
    "Endocrocin": ("NATURAL", False, "fungal anthraquinone"),
    "ent-Kaurene": ("NATURAL", False, "plant diterpene"),
    "Erucamide": ("PLASTIC", True, "polyolefin slip agent"),
    "Ethyl 2,4,8-tetradecatrienoate": ("NATURAL", False, "food-derived ester"),
    "Ethyl 2-acetyl-4,4,5,5-tetrafluoro-3-oxopentanoate": ("INDUSTRIAL", False, "fluorinated synthon"),
    "Galaxolidone": ("PPCP", True, "galaxolide (HHCB) oxidation TP, polycyclic musk"),
    "Genistein": ("NATURAL", False, "soy isoflavone"),
    "Grandiflorenic acid": ("NATURAL", False, "plant diterpenoid"),
    "Hexadecanamide": ("PLASTIC", False, "fatty amide slip agent / biogenic"),
    "Hexamethoxymethyl melamine": ("TRWP", True, "HMMM, tire crosslinker, road runoff tracer"),
    "Icaridin": ("PPCP", True, "picaridin insect repellent"),
    "ionene (associated with cocoa)": ("NATURAL", False, "food-derived terpenoid"),
    "Isoquinoline": ("INDUSTRIAL", True, "coal tar / combustion-related N-heterocycle"),
    "Lauryldimethylamine oxide": ("SURFACTANT", False, "amine oxide surfactant"),
    "Linoleic Acid": ("NATURAL", False, "plant fatty acid"),
    "Luteolin": ("NATURAL", False, "plant flavone"),
    "Mecoprop": ("PESTICIDE", True, "phenoxy herbicide, bitumen roofing-membrane leachate"),
    "Methyl (5ξ)-11,12-dihydroxyabieta-8,11,13-trien-20-oate": ("NATURAL", False, "abietane resin derivative"),
    "Methyl 9-tetradecenoate": ("NATURAL", False, "fatty acid methyl ester"),
    "MONOISONONYL PHOSPHATE": ("OPE", False, "alkyl phosphate"),
    "Monoolein": ("NATURAL", False, "monoacylglycerol"),
    "Myristyl sulfate": ("SURFACTANT", False, "alkyl sulfate surfactant"),
    "N,N-Dicyclohexylurea": ("TRWP", False, "rubber-related urea / synthesis byproduct"),
    "N,N-dimethyl-9H-purin-6-amine": ("INDUSTRIAL", False, "purine derivative"),
    "N,N-Dimethyldecylamine N-oxide": ("SURFACTANT", False, "amine oxide surfactant"),
    "N,N-Diphenylguanidine": ("TRWP", True, "DPG, vulcanisation accelerator, road runoff tracer"),
    "N,N-Diphenylurea": ("TRWP", True, "DPG transformation product"),
    "N-[2-(3-Phenylureido)phenyl]benzenesulfonamide": ("INDUSTRIAL", False, "sulfonamide-urea industrial chemical"),
    "Naringenin": ("NATURAL", False, "citrus flavanone"),
    "N-Benzyl-1-dodecanamine": ("SURFACTANT", False, "cationic surfactant"),
    "N-benzyl-N-methyl-N-phenylurea": ("INDUSTRIAL", False, "substituted urea"),
    "N-Ethyl-p-menthane-3-carboxamide": ("PPCP", False, "WS-3 cooling agent, personal care"),
    "Nobiletin": ("NATURAL", False, "citrus polymethoxyflavone"),
    "N-Octyl-2-pyrrolidone": ("INDUSTRIAL", False, "industrial solvent"),
    "NP-001057": ("UNKNOWN", False, "tentative identification, use unassigned"),
    "NP-002016": ("UNKNOWN", False, "tentative identification, use unassigned"),
    "NP-014113": ("UNKNOWN", False, "tentative identification, use unassigned"),
    "NP-016354": ("UNKNOWN", False, "tentative identification, use unassigned"),
    "NP-017061": ("UNKNOWN", False, "tentative identification, use unassigned"),
    "NS00004605": ("UNKNOWN", False, "tentative identification, use unassigned"),
    "O,O-Dibutyl phosphorothioate": ("OPE", True, "organothiophosphate, pesticide-related"),
    "Octadecanamine": ("SURFACTANT", False, "fatty amine"),
    "Octadecyl hydrogen sulfate": ("SURFACTANT", False, "alkyl sulfate surfactant"),
    "Octyl hydrogen phthalate": ("PLASTIC", True, "monoester phthalate TP"),
    "Oleamide": ("PLASTIC", True, "polyolefin slip agent"),
    "Oleanolic acid": ("NATURAL", False, "plant triterpenoid"),
    "o-Toluidine": ("INDUSTRIAL", True, "IARC Group 1 carcinogen, dye/rubber intermediate"),
    "Palmitoyl ethanolamide": ("NATURAL", False, "endogenous lipid"),
    "PEG n6": ("SURFACTANT", False, "polyethylene glycol oligomer"),
    "PEG n7": ("SURFACTANT", False, "polyethylene glycol oligomer"),
    "PEG n8": ("SURFACTANT", False, "polyethylene glycol oligomer"),
    "Perfluoro-1-octanesulfonic acid (PFOS)": ("PFAS", True, "POP, Stockholm Convention Annex B"),
    "Perfluoro-2-methyl-3-(propan-2-yl)pentan-3-yl": ("PFAS", True, "branched perfluorinated compound"),
    "Perfluorodecanoic acid (PFDA)": ("PFAS", True, "long-chain PFCA"),
    "Perfluorononanoic acid (PFNA)": ("PFAS", True, "long-chain PFCA"),
    "Perfluorooctanoic acid (PFOA)": ("PFAS", True, "POP, Stockholm Convention Annex A"),
    "Phthaldialdehyde": ("INDUSTRIAL", False, "disinfectant / derivatisation reagent"),
    "Piperine": ("NATURAL", False, "pepper alkaloid, food marker"),
    "PPG n4": ("SURFACTANT", False, "polypropylene glycol oligomer"),
    "PPG n7": ("SURFACTANT", False, "polypropylene glycol oligomer"),
    "PPG n8": ("SURFACTANT", False, "polypropylene glycol oligomer"),
    "Psoralen": ("NATURAL", False, "plant furanocoumarin"),
    "Scopoletin": ("NATURAL", False, "plant coumarin"),
    "Sodium [dodecanoyl(methyl)amino]acetate": ("SURFACTANT", False, "sodium lauroyl sarcosinate"),
    "Stearamide": ("PLASTIC", False, "fatty amide slip agent"),
    "STEARAMINE OXIDE": ("SURFACTANT", False, "amine oxide surfactant"),
    "Stearoyl Ethanolamide": ("NATURAL", False, "endogenous lipid"),
    "steviol": ("LIFESTYLE", False, "steviol glycoside aglycone, sweetener marker"),
    "Tangeritin": ("NATURAL", False, "citrus polymethoxyflavone"),
    "Taurochenodeoxycholic Acid (sodium salt)": ("LIFESTYLE", True, "conjugated bile acid, faecal marker"),
    "Testosterone sulfate": ("LIFESTYLE", True, "human steroid excretion marker"),
    "TETRADECYLDIETHANOLAMINE": ("SURFACTANT", False, "cationic/amphoteric surfactant"),
    "Tetraglyme": ("INDUSTRIAL", True, "glyme solvent, reprotoxic (SVHC)"),
    "trans-3-Hydroxycotinine": ("LIFESTYLE", True, "major nicotine metabolite"),
    "Tributylamine": ("INDUSTRIAL", False, "industrial amine"),
    "Trideca-2,4,7-trienal (flavoring)": ("NATURAL", False, "flavour aldehyde"),
    "Triethyl phosphate": ("OPE", True, "flame retardant / plasticiser"),
    "Triethylene glycol monobutyl ether": ("INDUSTRIAL", False, "glycol ether solvent"),
    "Triisopropanolamine": ("INDUSTRIAL", False, "cement grinding aid / corrosion inhibitor"),
    "Triphenylphosphine oxide": ("INDUSTRIAL", False, "ubiquitous synthesis byproduct"),
    "Tris(2-butoxyethyl) phosphate": ("OPE", True, "TBOEP, flame retardant, floor polish"),
    "Tris(2-ethylhexyl) phosphate": ("OPE", True, "TEHP, flame retardant / plasticiser"),
    "Venlafaxine": ("PPCP", True, "antidepressant, wastewater marker"),
    "Voriconazole": ("PPCP", True, "antifungal pharmaceutical"),
    "vulpinic acid": ("NATURAL", False, "lichen metabolite"),
    "Xanthurenic acid": ("NATURAL", False, "tryptophan metabolite"),
    "4-Aminodiphenylamine": ("TRWP", True, "6PPD precursor / transformation product"),
    "Costunolide": ("NATURAL", False, "sesquiterpene lactone"),
    "Cytisinicline": ("PPCP", False, "cytisine, smoking-cessation alkaloid"),
    "Ibuprofen": ("PPCP", True, "NSAID, wastewater marker"),
    "Lauramidopropyl_betaine": ("SURFACTANT", False, "amphoteric surfactant"),
    "N-(2,4-Dimethylphenyl)acetamide": ("INDUSTRIAL", False, "acetamide, pesticide-related intermediate"),
    "N,N-Bis(2-hydroxyethyl)dodecanamide": ("SURFACTANT", False, "lauramide DEA surfactant"),
    "Triethylene glycol dimethyl ether": ("INDUSTRIAL", True, "triglyme solvent, reprotoxic (SVHC)"),
    "Formylated HMMM": ("TRWP", True, "HMMM transformation product, road runoff tracer"),
}


def annotate(compound_names):
    """Return (classes, priority_flags, notes) aligned to `compound_names`."""
    classes, prio, notes = [], [], []
    for name in compound_names:
        c, p, n = ANNOTATION.get(name, ("UNKNOWN", False, "not in annotation table"))
        classes.append(c)
        prio.append(p)
        notes.append(n)
    return classes, prio, notes
