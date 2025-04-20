# Imports packages
import sys
import sys #reimport sys so we have it when not running package installation/setup
import infofile # local file containing cross-sections, sums of weights, dataset IDs
import numpy as np # for numerical calculations such as histogramming

import uproot # for reading .root files
import awkward as ak # to represent nested data in columnar format
import vector # for 4-momentum calculations
import time
import pickle


# Creates a dictionary of all the data
samples = {
    'data': {
        'list' : ['data_A','data_B','data_C','data_D'], # data is from 2016, first four periods of data taking (ABCD)
    },
    'top':{ #processes with a top quark (single top, ttbar)
        'list' : ['ttbar_lep','single_top_tchan','single_antitop_tchan','single_top_wtchan','single_antitop_wtchan',
                  'single_top_schan','single_antitop_schan','ttW','ttee','ttmumu'],
        'color' : "#ffb255",
        'legend' : r'Top processes',
    },  
}

# Prompts the user for a process to work with
print("Please select a process to work with:")
for i in range(len(samples["top"]["list"])):
    print(f"{i}: {samples['top']['list'][i]}")

process_num = int(input("Enter the number corresponding to the process: "))

if process_num < 0 or process_num >= len(samples['top']['list']):
    print("Invalid selection. Please try again.")
    sys.exit(1)
    
# Prompts the user for a cut value for the b-jet selection.
# can be cut70, cut77 or cut85
cut_value = input("Please enter a cut value for the b-jet selection (cut70, cut77, cut85): ").strip()
if cut_value not in ['cut70', 'cut77', 'cut85']:
    print("Invalid input. Please enter cut70, cut77 or cut85.")
    sys.exit(1)

# Prompts the user for a filename to save to
filename = input("Please enter a filename to save the data to (without .pkl): ").strip()

# Skims data
# Defines path to open data
path      = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/"

value = samples["top"]["list"][process_num] # Gets the value of the process selected by the user
# Retrieves the root file
background_top_path = path + "1lep/MC/mc_"+str(infofile.infos[value]["DSID"])+"."+value+".1lep.root"

# Loads in the data
tree = uproot.open(background_top_path + ":mini")

print("WARNING: reading data can take up to 15 minutes")
print("Reading data...")
# Grabs the data for the jets
b_jet_selection = tree.arrays("jet_MV2c10", library="ak")["jet_MV2c10"]
jet_pt          = tree.arrays("jet_pt", library="ak")["jet_pt"]
jet_eta         = tree.arrays('jet_eta', library="ak")['jet_eta']
jet_E           = tree.arrays('jet_E', library="ak")['jet_E']
jet_phi         = tree.arrays("jet_phi", library="ak")["jet_phi"]
print("jets complete")

# Gets the data for the leptons
lep_type = tree.arrays("lep_type", library="ak")["lep_type"]
lep_pt   = tree.arrays("lep_pt", library="ak")["lep_pt"]
lep_eta  = tree.arrays("lep_eta", library="ak")["lep_eta"]
lep_phi  = tree.arrays("lep_phi", library="ak")["lep_phi"]
lep_E    = tree.arrays("lep_E", library="ak")["lep_E"]
print("leptons complete")

# Gets the missing transverse energy
met_Et  = tree.arrays("met_et", library="ak")["met_et"]
met_phi = tree.arrays("met_phi", library="ak")["met_phi"]
print("missing Et complete")
print("Reading data complete")

# Sets up the weight variables
mcWeight               = tree.arrays("mcWeight", library="ak")["mcWeight"]
scaleFactor_PILEUP     = tree.arrays("scaleFactor_PILEUP", library="ak")["scaleFactor_PILEUP"]
scaleFactor_ELE        = tree.arrays("scaleFactor_ELE", library="ak")["scaleFactor_ELE"]
scaleFactor_MUON       = tree.arrays("scaleFactor_MUON", library="ak")["scaleFactor_MUON"]
scaleFactor_LepTRIGGER = tree.arrays("scaleFactor_LepTRIGGER", library="ak")["scaleFactor_LepTRIGGER"]

################################################################################################
print("Applying Cuts...")
# First look at what events actually have a b-jet present. Different cuts can be used to get different confidences, though at the cost of 
# removal of a number of events.
cut70 = 0.8244
cut77 = 0.6459
cut85 = 0.1758
cut_dict = {}
cut_dict['cut70'] = cut70
cut_dict['cut77'] = cut77
cut_dict['cut85'] = cut85
cut = cut_dict[cut_value] # Selects the cut value based on user input

# Generates the mask for events with exactly 2 jets
jet_mask = ak.num(jet_pt, axis=1) == 2

# Applies the mask to the jet data
b_jet_selection_cut = b_jet_selection[jet_mask]
jet_pt_cut          = jet_pt[jet_mask]
lep_type_cut        = lep_type[jet_mask]
lep_pt_cut          = lep_pt[jet_mask]
lep_eta_cut         = lep_eta[jet_mask]
jet_eta_cut         = jet_eta[jet_mask]
met_Et_cut          = met_Et[jet_mask]
lep_phi_cut         = lep_phi[jet_mask]
met_phi_cut         = met_phi[jet_mask]
lep_E_cut           = lep_E[jet_mask]
jet_E_cut           = jet_E[jet_mask]
jet_phi_cut         = jet_phi[jet_mask]

# Applies the cut to the weight variables
mcWeight_cut               = mcWeight[jet_mask]
scaleFactor_PILEUP_cut     = scaleFactor_PILEUP[jet_mask]
scaleFactor_ELE_cut        = scaleFactor_ELE[jet_mask]
scaleFactor_MUON_cut       = scaleFactor_MUON[jet_mask]
scaleFactor_LepTRIGGER_cut = scaleFactor_LepTRIGGER[jet_mask]

print(f'Length of data before 2-jet check {len(lep_pt):0.2e}')
print(f'Length of data after 2-jet check {len(lep_pt_cut):0.2e}')
print("")

#########################################################################################################
# Applies the jet selection to determine if a b-jet is present.
jet_mask2  = ak.Array(b_jet_selection_cut > cut) # Determines where the value is above selection cut
jet_mask2  = ak.values_astype(jet_mask2, "int")    # sets the truth values to binary
jet_mask2  = ak.sum(jet_mask2, axis=1)             # Finds the events where there is a b-jet as determined from above
jet_mask2  = ak.where(jet_mask2 > 1, 0, jet_mask2) # Ensures there is only one jet
jet_mask2  = ak.values_astype(jet_mask2, "bool")   # Returns the value to a boolean

print(f'Length of data before b-jet discrimination {len(lep_pt_cut):0.2e}')
# Applies the mask on each of the data types
b_jet_selection_cut = b_jet_selection_cut[jet_mask2]
jet_pt_cut          = jet_pt_cut[jet_mask2]
lep_pt_cut          = lep_pt_cut[jet_mask2]
lep_eta_cut         = lep_eta_cut[jet_mask2]
jet_eta_cut         = jet_eta_cut[jet_mask2]
met_Et_cut          = met_Et_cut[jet_mask2]
lep_phi_cut         = lep_phi_cut[jet_mask2]
met_phi_cut         = met_phi_cut[jet_mask2]
lep_E_cut           = lep_E_cut[jet_mask2]
jet_E_cut           = jet_E_cut[jet_mask2]
jet_phi_cut         = jet_phi_cut[jet_mask2]

# Applies the cut to the weight variables
mcWeight_cut               = mcWeight_cut[jet_mask2]
scaleFactor_PILEUP_cut     = scaleFactor_PILEUP_cut[jet_mask2]
scaleFactor_ELE_cut        = scaleFactor_ELE_cut[jet_mask2]
scaleFactor_MUON_cut       = scaleFactor_MUON_cut[jet_mask2]
scaleFactor_LepTRIGGER_cut = scaleFactor_LepTRIGGER_cut[jet_mask2]

print(f'Length of data after b-jet discrimination {len(lep_pt_cut):0.2e}')
print("")

#########################################################################################################
# Sets up a cut such that the pseudorapidity of the untagged jet is greater than 1.5
eta_jet_thresh  = 1.5
b_jet_mask      = ak.Array(b_jet_selection_cut > cut)

untagged_jets_eta = jet_eta_cut[~b_jet_mask]
keep_mask         = ak.all(untagged_jets_eta > eta_jet_thresh, axis=1)

print(f'Length of data before untagged jet eta cut {len(lep_pt_cut):0.2e}')
# Applies the mask to each data type
b_jet_selection_cut = b_jet_selection_cut[keep_mask]
jet_pt_cut          = jet_pt_cut[keep_mask]
lep_pt_cut          = lep_pt_cut[keep_mask]
lep_eta_cut         = lep_eta_cut[keep_mask]
jet_eta_cut         = jet_eta_cut[keep_mask]
met_Et_cut          = met_Et_cut[keep_mask]
lep_phi_cut         = lep_phi_cut[keep_mask]
met_phi_cut         = met_phi_cut[keep_mask]
lep_E_cut           = lep_E_cut[keep_mask]
jet_E_cut           = jet_E_cut[keep_mask]
jet_phi_cut         = jet_phi_cut[keep_mask]

# Applies the cut to the weight variables
mcWeight_cut               = mcWeight_cut[keep_mask]
scaleFactor_PILEUP_cut     = scaleFactor_PILEUP_cut[keep_mask]
scaleFactor_ELE_cut        = scaleFactor_ELE_cut[keep_mask]
scaleFactor_MUON_cut       = scaleFactor_MUON_cut[keep_mask]
scaleFactor_LepTRIGGER_cut = scaleFactor_LepTRIGGER_cut[keep_mask]

print(f'Length of data after untagged jet eta cut {len(lep_pt_cut):0.2e}')
print("")

#########################################################################################################
# Sets up a cut such that the difference in pseudorapidity between any b-jets and untagged jets is > 1.5
eta_diff_thresh  = 1.5
b_jet_mask       = ak.Array(b_jet_selection_cut > cut)

# Separate out the b-jets' eta and the untagged jets' eta
b_jets_eta        = jet_eta_cut[b_jet_mask]
untagged_jets_eta = jet_eta_cut[~b_jet_mask]

# Form all b-jet ↔ untagged-jet pairs for each event
pairs     = ak.cartesian([b_jets_eta, untagged_jets_eta], axis=1)
delta_eta = pairs["1"] - pairs["0"]

# Keep only events where *every* b-untagged pair has |eta_b - eta_u| > 1.5
keep_mask2 = ak.all(delta_eta > eta_diff_thresh, axis=1)

print(f'Length of data before jet eta diff cut {len(lep_pt_cut):0.2e}')
# Applies the event-level mask to each data array
b_jet_selection_cut = b_jet_selection_cut[keep_mask2]
jet_pt_cut          = jet_pt_cut[keep_mask2]
lep_pt_cut          = lep_pt_cut[keep_mask2]
lep_eta_cut         = lep_eta_cut[keep_mask2]
jet_eta_cut         = jet_eta_cut[keep_mask2]
met_Et_cut          = met_Et_cut[keep_mask2]
lep_phi_cut         = lep_phi_cut[keep_mask2]
met_phi_cut         = met_phi_cut[keep_mask2]
lep_E_cut           = lep_E_cut[keep_mask2]
jet_E_cut           = jet_E_cut[keep_mask2]
jet_phi_cut         = jet_phi_cut[keep_mask2]

# Applies the cut to the weight variables
mcWeight_cut               = mcWeight_cut[keep_mask2]
scaleFactor_PILEUP_cut     = scaleFactor_PILEUP_cut[keep_mask2]
scaleFactor_ELE_cut        = scaleFactor_ELE_cut[keep_mask2]
scaleFactor_MUON_cut       = scaleFactor_MUON_cut[keep_mask2]
scaleFactor_LepTRIGGER_cut = scaleFactor_LepTRIGGER_cut[keep_mask2]

print(f'Length of data after jet eta diff cut {len(lep_pt_cut):0.2e}')
print("")

#########################################################################################################
# Sets up a cut such that the minimum jet transverse momentum is greater than 30GeV
pt_jet_thresh  = 30 #GeV
pt_thresh_mask = (jet_pt_cut[:,0]*1e-3) > pt_jet_thresh

print(f'Length of data before b-jet pt cut {len(lep_pt_cut):0.2e}')
# Applies the mask to each data type
b_jet_selection_cut = b_jet_selection_cut[pt_thresh_mask]
jet_pt_cut          = jet_pt_cut[pt_thresh_mask]
lep_pt_cut          = lep_pt_cut[pt_thresh_mask]
lep_eta_cut         = lep_eta_cut[pt_thresh_mask]
jet_eta_cut         = jet_eta_cut[pt_thresh_mask]
met_Et_cut          = met_Et_cut[pt_thresh_mask]
lep_phi_cut         = lep_phi_cut[pt_thresh_mask]
met_phi_cut         = met_phi_cut[pt_thresh_mask]
lep_E_cut           = lep_E_cut[pt_thresh_mask]
jet_E_cut           = jet_E_cut[pt_thresh_mask]
jet_phi_cut         = jet_phi_cut[pt_thresh_mask]

# Applies the cut to the weight variables
mcWeight_cut               = mcWeight_cut[pt_thresh_mask]
scaleFactor_PILEUP_cut     = scaleFactor_PILEUP_cut[pt_thresh_mask]
scaleFactor_ELE_cut        = scaleFactor_ELE_cut[pt_thresh_mask]
scaleFactor_MUON_cut       = scaleFactor_MUON_cut[pt_thresh_mask]
scaleFactor_LepTRIGGER_cut = scaleFactor_LepTRIGGER_cut[pt_thresh_mask]

print(f'Length of data after b-jet pt cut {len(lep_pt_cut):0.2e}')
print("")

#########################################################################################################
# Cuts events with a transveres momentum of the lepton to be less than 35 GeV
lep_mom_cut = ak.Array((lep_pt_cut*1e-3) > 35)
lep_mom_cut = ak.values_astype(lep_mom_cut, "int")
lep_mom_cut = ak.sum(lep_mom_cut, axis=1)             
lep_mom_cut = ak.values_astype(lep_mom_cut, "bool")   # Returns the value to a boolean

print(f'Length of data before lepton momentum cut {len(lep_pt_cut):0.2e}')
# # Applies the mask to all the data
b_jet_selection_cut = b_jet_selection_cut[lep_mom_cut]
jet_pt_cut          = jet_pt_cut[lep_mom_cut]
lep_pt_cut          = lep_pt_cut[lep_mom_cut]
lep_eta_cut         = lep_eta_cut[lep_mom_cut]
jet_eta_cut         = jet_eta_cut[lep_mom_cut]
met_Et_cut          = met_Et_cut[lep_mom_cut]
lep_phi_cut         = lep_phi_cut[lep_mom_cut]
met_phi_cut         = met_phi_cut[lep_mom_cut]
lep_E_cut           = lep_E_cut[lep_mom_cut]
jet_E_cut           = jet_E_cut[lep_mom_cut]
jet_phi_cut         = jet_phi_cut[lep_mom_cut]

# Applies the cut to the weight variables
mcWeight_cut               = mcWeight_cut[lep_mom_cut]
scaleFactor_PILEUP_cut     = scaleFactor_PILEUP_cut[lep_mom_cut]
scaleFactor_ELE_cut        = scaleFactor_ELE_cut[lep_mom_cut]
scaleFactor_MUON_cut       = scaleFactor_MUON_cut[lep_mom_cut]
scaleFactor_LepTRIGGER_cut = scaleFactor_LepTRIGGER_cut[lep_mom_cut]

print(f'Length of data after lepton momentum cut {len(lep_pt_cut):0.2e}')
print("")

#########################################################################################################
# Sets up the missing pt cut
miss_Et_cut_val  = 30 #GeV
miss_Et_cut_mask = ak.Array((met_Et_cut * 1e-3) > miss_Et_cut_val)

print(f'Length of data before missing energy cut {len(lep_pt_cut):0.2e}')
# # Applies the mask to all the data
b_jet_selection_cut = b_jet_selection_cut[miss_Et_cut_mask]
jet_pt_cut          = jet_pt_cut[miss_Et_cut_mask]
lep_pt_cut          = lep_pt_cut[miss_Et_cut_mask]
lep_eta_cut         = lep_eta_cut[miss_Et_cut_mask]
jet_eta_cut         = jet_eta_cut[miss_Et_cut_mask]
met_Et_cut          = met_Et_cut[miss_Et_cut_mask]
lep_phi_cut         = lep_phi_cut[miss_Et_cut_mask]
met_phi_cut         = met_phi_cut[miss_Et_cut_mask]
lep_E_cut           = lep_E_cut[miss_Et_cut_mask]
jet_E_cut           = jet_E_cut[miss_Et_cut_mask]
jet_phi_cut         = jet_phi_cut[miss_Et_cut_mask]

# Applies the cut to the weight variables
mcWeight_cut               = mcWeight_cut[miss_Et_cut_mask]
scaleFactor_PILEUP_cut     = scaleFactor_PILEUP_cut[miss_Et_cut_mask]
scaleFactor_ELE_cut        = scaleFactor_ELE_cut[miss_Et_cut_mask]
scaleFactor_MUON_cut       = scaleFactor_MUON_cut[miss_Et_cut_mask]
scaleFactor_LepTRIGGER_cut = scaleFactor_LepTRIGGER_cut[miss_Et_cut_mask]

print(f'Length of data after missing energy cut {len(lep_pt_cut):0.2e}')
print("")

#########################################################################################################
# calculates the momenta values to be used
pt1x = lep_pt_cut*np.cos(lep_phi_cut)
pt1y = lep_pt_cut*np.sin(lep_phi_cut)
pt2x = np.sqrt(met_Et_cut)*np.cos(met_phi_cut)
pt2y = np.sqrt(met_Et_cut)*np.sin(met_phi_cut)

# Adds the momentum components and computes the square
px = pt1x + pt2x
py = pt1y + pt2y
p_sq = px**2 + py**2

# Computes the energy term
E_sq = lep_E_cut**2 + met_Et_cut**2

# Computes the transverse mass
MtW = np.sqrt(E_sq + p_sq)

# APPLIES THE TRANSVERSE MASS CUT
mass_T_cut = 60 #GeV
mask7 = (MtW * 1e-3) > mass_T_cut
mask7 = mask7[:,0]

print(f'Length of data before transverse mass cut {len(lep_pt_cut):0.2e}')
b_jet_selection_cut = b_jet_selection_cut[mask7]
jet_pt_cut          = jet_pt_cut[mask7]
lep_pt_cut          = lep_pt_cut[mask7]
lep_eta_cut         = lep_eta_cut[mask7]
jet_eta_cut         = jet_eta_cut[mask7]
met_Et_cut          = met_Et_cut[mask7]
lep_phi_cut         = lep_phi_cut[mask7]
met_phi_cut         = met_phi_cut[mask7]
lep_E_cut           = lep_E_cut[mask7]
jet_E_cut           = jet_E_cut[mask7]
jet_phi_cut         = jet_phi_cut[mask7]
MtW_cut             = MtW[mask7]

# Applies the cut to the weight variables
mcWeight_cut               = mcWeight_cut[mask7]
scaleFactor_PILEUP_cut     = scaleFactor_PILEUP_cut[mask7]
scaleFactor_ELE_cut        = scaleFactor_ELE_cut[mask7]
scaleFactor_MUON_cut       = scaleFactor_MUON_cut[mask7]
scaleFactor_LepTRIGGER_cut = scaleFactor_LepTRIGGER_cut[mask7]

print(f'Length of data after transverse mass cut {len(lep_pt_cut):0.2e}')
print("")

#########################################################################################################
# Calculates H_T
HT = ak.sum(jet_pt_cut, axis=1) + lep_pt_cut + met_Et_cut

# Applies the scalar sum cut
HT_min = 195 #GeV
mask8  = (HT * 1e-3) > HT_min 
mask8  = mask8[:,0]

print(f'Length of data scalar sum cut {len(lep_pt_cut):0.2e}')
b_jet_selection_cut = b_jet_selection_cut[mask8]
jet_pt_cut          = jet_pt_cut[mask8]
lep_pt_cut          = lep_pt_cut[mask8]
lep_eta_cut         = lep_eta_cut[mask8]
jet_eta_cut         = jet_eta_cut[mask8]
met_Et_cut          = met_Et_cut[mask8]
lep_phi_cut         = lep_phi_cut[mask8]
met_phi_cut         = met_phi_cut[mask8]
lep_E_cut           = lep_E_cut[mask8]
jet_E_cut           = jet_E_cut[mask8]
jet_phi_cut         = jet_phi_cut[mask8]
MtW_cut             = MtW_cut[mask8]
HT_cut              = HT[mask8]

# Applies the cut to the weight variables
mcWeight_cut               = mcWeight_cut[mask8]
scaleFactor_PILEUP_cut     = scaleFactor_PILEUP_cut[mask8]
scaleFactor_ELE_cut        = scaleFactor_ELE_cut[mask8]
scaleFactor_MUON_cut       = scaleFactor_MUON_cut[mask8]
scaleFactor_LepTRIGGER_cut = scaleFactor_LepTRIGGER_cut[mask8]

print(f'Length of data after scalar sum cut {len(lep_pt_cut):0.2e}')
print("")

#########################################################################################################
# Generates a mask to only get the b tagged jet
b_jet_mask = ak.Array(b_jet_selection_cut > cut)

# Computes the energy term
E_sq = (lep_E_cut + jet_E_cut[b_jet_mask])**2

# Computes the momentum term
# lepton
p1x = lep_pt_cut * np.cos(lep_phi_cut)
p1y = lep_pt_cut * np.sin(lep_phi_cut)
p1z = lep_pt_cut * np.sinh(lep_eta_cut)
# b-jet
p2x = jet_pt_cut[b_jet_mask] * np.cos(jet_phi_cut[b_jet_mask])
p2y = jet_pt_cut[b_jet_mask] * np.sin(jet_phi_cut[b_jet_mask])
p2z = jet_pt_cut[b_jet_mask] * np.sinh(jet_eta_cut[b_jet_mask])
# Adds the momentum together
p_totx = p1x + p2x
p_toty = p1y + p2y
p_totz = p1z + p2z
# Computes the square
p_sq = p_totx**2 + p_toty**2 + p_totz**2

# Computes the mass
m_lb = np.sqrt(E_sq + p_sq)

# Applies the cut for the mass
min_mass = 150 # GeV
mask9  = (m_lb * 1e-3) > min_mass
mask9  = mask9[:,0]

print(f'Length of data before lepton b-jet mass cut {len(lep_pt_cut):0.2e}')
b_jet_selection_cut = b_jet_selection_cut[mask9]
jet_pt_cut          = jet_pt_cut[mask9]
lep_pt_cut          = lep_pt_cut[mask9]
lep_eta_cut         = lep_eta_cut[mask9]
jet_eta_cut         = jet_eta_cut[mask9]
met_Et_cut          = met_Et_cut[mask9]
lep_phi_cut         = lep_phi_cut[mask9]
met_phi_cut         = met_phi_cut[mask9]
lep_E_cut           = lep_E_cut[mask9]
jet_E_cut           = jet_E_cut[mask9]
jet_phi_cut         = jet_phi_cut[mask9]
MtW_cut             = MtW_cut[mask9]
HT_cut              = HT[mask9]
m_lb_cut            = m_lb[mask9]

# Applies the cut to the weight variables
mcWeight_cut               = mcWeight_cut[mask9]
scaleFactor_PILEUP_cut     = scaleFactor_PILEUP_cut[mask9]
scaleFactor_ELE_cut        = scaleFactor_ELE_cut[mask9]
scaleFactor_MUON_cut       = scaleFactor_MUON_cut[mask9]
scaleFactor_LepTRIGGER_cut = scaleFactor_LepTRIGGER_cut[mask9]


print(f'Length of data after lepton b-jet mass cut {len(lep_pt_cut):0.2e}')
print("")

#########################################################################################################
# Computes the total weight for each event
total_weight = mcWeight_cut * scaleFactor_PILEUP_cut * scaleFactor_ELE_cut * scaleFactor_MUON_cut * scaleFactor_LepTRIGGER_cut
print("Computed total weight not multiplied by xsec_weight")

# Saves the data to be called in a different notebook
with open(filename+".pkl", "wb") as f:
    pickle.dump([b_jet_selection_cut, jet_pt_cut, lep_pt_cut, lep_eta_cut, jet_eta_cut, met_Et_cut, lep_phi_cut, 
                 met_phi_cut, lep_E_cut, jet_E_cut, jet_phi_cut, MtW_cut, HT_cut, m_lb_cut, total_weight], f)
    
print(f"Data saved to {filename}.pkl")
print("Done!")