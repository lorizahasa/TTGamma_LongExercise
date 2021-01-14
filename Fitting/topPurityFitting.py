from ROOT import TFile, TFractionFitter, TObjArray

import pprint

## open root file, created from the SaveHistogramsToRoot step, containing M3 distributions 
_file = TFile("../RootFiles/M3_Output.root")

## List of systematics
systematics  = ["nominal",
                "FSRDown",
                "FSRUp",
                "ISRDown",
                "ISRUp",
                "JERDown",
                "JERUp",
                "JESDown",
                "JESUp",
                "PDFDown",
                "PDFUp",
                "Q2ScaleDown",
                "Q2ScaleUp",
                "btagWeight",
                "btagWeight",
                "eleEffWeightDown",
                "eleEffWeightUp",
                "muEffWeightDown",
                "muEffWeightUp",
                "puWeightDown",
                "puWeightUp",
]

results = {}

## Get data from the input root file
data = _file.Get("dataObs")

## Loop over the list of systematics
for syst in systematics:
    
    ## Create an array 'mc' of histograms from TopPair and NonTop categories
    mc = TObjArray(2)   
    ## Add TopPair and NonTop histograms to the array 'mc'
    mc.Add(_file.Get("TopPair_"+syst))
    mc.Add(_file.Get("NonTop_"+syst))
    
       if not _file.Get("TopPair_"+syst):
        print(syst)
        continue
    
    ## Fit the MC histograms to data 
    fit = TFractionFitter(data, mc, "q")
    
    ## fit.Fit() actually performs the fit
    ## check the fit status
    status = int(fit.Fit())
    
    ## status==0 corresponds to fits that converged
    ## Now we can extract value of topPurity (fraction of events coming from top pair production) and the error on that value, topPurityErr for each systematic
    if not status==0:
        print (f"Error in fit while processing {syst} sample: exit status {status}")
        
    ## Get the value of fit parameter and its error for the TopPair MC category: 
    topPurity = fit.GetFitter().Result().Parameters()[0]
    topPurityErr = fit.GetFitter().Result().ParError(0)
    
    ## Fill the dictionary "results" with the topPurity and topPurityErr for each systematic
    results[syst] = (topPurity, topPurityErr)

    del fit


pp = pprint.PrettyPrinter(indent=4)
pprint.pprint(results)
