import uproot
uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
from coffea import util, processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from ttgamma import TTGammaProcessor
#from ttgamma.utils.fileSet_2016_LZ4 import fileSet_2016 as fileset
#from ttgamma.utils.fileSet_2016_LZ4 import fileSet_Data_2016
from ttgamma.utils.fileset2021 import fileset
from ttgamma.utils.crossSections import *

import time
import sys
from pprint import pprint

import argparse
parser = argparse.ArgumentParser(description="Batch processing script for ttgamma analysis")
parser.add_argument("mcGroup", type=str, help="Name of process to run (Data, MCTTGamma, MCTTbar1l, MCTTbar2l, MCSingleTop, MCZJets, MCWJets, MCOther)")
parser.add_argument("--chunksize", type=int, default=100000, help="Chunk size")
parser.add_argument("--maxchunks", type=int, default=None, help="Max chunks")
parser.add_argument("--workers", type=int, default=1, help="Number of workers")
parser.add_argument("--condor", action="store_true", help="Flag for running on condor (disables progress bar)")
args = parser.parse_args()

tstart = time.time()

print("Running {}".format(args.mcGroup))

#job_fileset = {args.mcGroup: fileset[args.mcGroup]} #{key: fileset[key] for key in fileset if "Data" in key}

if args.mcGroup == "Data":
    job_fileset = {key: fileset[key] for key in fileset if "Data" in key}
    output = processor.run_uproot_job(job_fileset,
                                      treename           = 'Events',
                                      processor_instance = TTGammaProcessor(isMC=False),
                                      executor           = processor.futures_executor,
                                      executor_args      = {'schema': NanoAODSchema, 'workers': args.workers, 'status': not args.condor},#{'workers': 4, 'flatten': True},
                                      chunksize          = args.chunksize,
                                      maxchunks          = args.maxchunks
                                  )
    
    elapsed = time.time() - tstart
    print("Total time: %.1f seconds"%elapsed)
    print("Total rate: %.1f events / second"%(output['EventCount'].value/elapsed))
    
    util.save(output, 'outputData_ttgamma_condorFull_4jet.coffea')

else:
    '''
    mcEventYields = {
      'DYjetsM10to50_2016' : 35114961.0, 'DYjetsM50_2016' : 146280395.0, 'GJets_HT40To100_2016' : 9326139.0, 'GJets_HT100To200_2016' : 10104155.0, 'GJets_HT200To400_2016' : 20527506.0, 'GJets_HT400To600_2016' : 5060070.0, 'GJets_HT600ToInf_2016' : 5080857.0, 'QCD_Pt20to30_Ele_2016' : 9241500.0, 'QCD_Pt30to50_Ele_2016' : 11508842.0, 'QCD_Pt50to80_Ele_2016' : 45789059.0, 'QCD_Pt80to120_Ele_2016' : 77800204.0, 'QCD_Pt120to170_Ele_2016' : 75367655.0, 'QCD_Pt170to300_Ele_2016' : 11105095.0, 'QCD_Pt300toInf_Ele_2016' : 7090318.0, 'QCD_Pt20to30_Mu_2016'      : 31878740.0, 'QCD_Pt30to50_Mu_2016'      : 29936360.0, 'QCD_Pt50to80_Mu_2016'      : 19662175.0, 'QCD_Pt80to120_Mu_2016'     : 23686772.0, 'QCD_Pt120to170_Mu_2016'    : 7897731.0, 'QCD_Pt170to300_Mu_2016'    : 17350231.0, 'QCD_Pt300to470_Mu_2016'    : 49005976.0, 'QCD_Pt470to600_Mu_2016'    : 19489276.0, 'QCD_Pt600to800_Mu_2016'    : 9981311.0, 'QCD_Pt800to1000_Mu_2016'   : 19940747.0, 'QCD_Pt1000toInf_Mu_2016'   : 13608903.0, 'ST_s_channel_2016'         : 6137801.0, 'ST_tW_channel_2016'        : 4945734.0, 'ST_tbarW_channel_2016'     : 4942374.0, 'ST_tbar_channel_2016'      : 17780700.0, 'ST_t_channel_2016'         : 31848000.0, 'TTGamma_Dilepton_2016'     : 5728644.0, 'TTGamma_Hadronic_2016'     : 5635346.0, 'TTGamma_SingleLept_2016'   : 10991612.0, 'TTWtoLNu_2016'             : 2716249.0, 'TTWtoQQ_2016'              : 430310.0, 'TTZtoLL_2016'              : 6420825.0, 'TTbarPowheg_Dilepton_2016' : 67339946.0, 'TTbarPowheg_Hadronic_2016' : 67963984.0, 'TTbarPowheg_Semilept_2016' : 106438920.0, 'W1jets_2016'               : 45283121.0, 'W2jets_2016'               : 60438768.0, 'W3jets_2016'               : 59300029.0, 'W4jets_2016' : 29941394.0, 'WGamma_01J_5f_2016' : 6103817.0, 'ZGamma_01J_5f_lowMass_2016': 9696539.0, 'WW_2016' : 7982180.0, 'WZ_2016' : 3997571.0, 'ZZ_2016' : 1988098.0
      }
    '''

    #if args.mcGroup == "MC":
    #    job_fileset = {key: fileset[key] for key in fileset if not "Data" in key}
    #    mcType = "MC"

    # Define mapping for running on condor
    mc_group_mapping = {
        "MCTTGamma": [key for key in fileset if "TTGamma" in key], 
        "MCTTbar1l": ["TTbarPowheg_Semilept", "TTbarPowheg_Hadronic"],
        "MCTTbar2l": ["TTbarPowheg_Dilepton"],
        "MCSingleTop": [key for key in fileset if "ST" in key],
        "MCZJets": [key for key in fileset if "DY" in key],
        "MCWJets": [key for key in fileset if "W1" in key or "W2" in key or "W3" in key or "W4" in key],
    }
    mc_nonother = []
    for mcType, sampleList in mc_group_mapping.items():
        mc_nonother.extend(sampleList)
    mc_group_mapping["MCOther"] = [key for key in fileset if (not key in mc_nonother) and (not "Data" in key)]
    job_fileset = {key: fileset[key] for key in mc_group_mapping[args.mcGroup]}

    pprint(job_fileset)

    output = processor.run_uproot_job(job_fileset,
                                      treename           = 'Events',
                                      processor_instance = TTGammaProcessor(isMC=True),
                                      executor           = processor.futures_executor, #processor.futures_executor,
                                      executor_args      = {'schema': NanoAODSchema, 'workers': args.workers, 'status': not args.condor},#{'workers': 4, 'flatten': True},
                                      chunksize          = args.chunksize,
                                      maxchunks          = args.maxchunks
                                  )

    elapsed = time.time() - tstart
    print("Total time: %.1f seconds"%elapsed)
    print("Total rate: %.1f events / second"%(output['EventCount'].value/elapsed))

    #util.save(output, f"output{args.mcGroup}_ttgamma_condorFull_4jet.coffea")

    # Compute original number of events for normalization
    output['InputEventCount'] = processor.defaultdict_accumulator(int)
    lumi_sfs = {}
    for dataset_name, dataset_files in job_fileset.items():
        for filename in dataset_files:
            with uproot.open(filename) as fhandle:
                output['InputEventCount'][dataset_name] +=fhandle["hEvents"].values()[2] - fhandle["hEvents"].values()[0]

        # Calculate luminosity scale factor
        lumi_sfs[dataset_name] = crossSections[dataset_name] * lumis[2016] / output["InputEventCount"][dataset_name]

    for key, obj in output.items():
        if isinstance(obj, hist.Hist):
            obj.scale(lumi_sfs, axis="dataset")
    util.save(output, f"output{args.mcGroup}_ttgamma_condorFull_4jet.coffea")



