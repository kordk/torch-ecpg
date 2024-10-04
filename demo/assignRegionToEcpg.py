#!/usr/bin/env python3

## kord.kober@ucsf.edu
## github.com/kordk/torch-ecpg

import os,sys,subprocess,time,datetime
import re,getopt,string
import numpy as np

DEBUG=0

## DEFAULTS - Kennedy et al. BMC Genomics (2018) 19:476

#PVALCUTOFF=0.00001                   ## 10-5 is "suggestive" in Kennedy 2018
#PVALCUTOFF=0.00000000001             ## 10-11 is "significant" in Kennedy 2018
PVALCUTOFF=np.float32(0.000001)       ## 10-6 is our "exploratory" cutoff

## DISTAL >50Kb TSS
DISTAL_OFFSET=50000

## CIS <50Kb TSS
CIS_OFFSET=0
CIS_UPSTREAM_DISTANCE=50000

## PROMOTER +/- 2500 bp TSS
PROMOTER_OFFSET=0
PROMOTER_UPSTREAM_DISTANCE=2500
PROMOTER_DOWNSTREAM_DISTANCE=2500

#### Read in the bed6 annotation to a dictionary #######################################
def readBed6AnnotatioFileToDict(my_bed6AnnotFile):
    my_lociH = {}

    bedFP = open(my_bed6AnnotFile, "r")

    # chrom  chromStart  chromEnd  name        score  strand
    # 1      805541      805541    cg16619049  0      +
    # 1      805554      805554    cg23100540  0      +
    # 1      812539      812539    cg18147296  0      +
    # 1      834183      834183    cg13938959  0      -
    # 1      834295      834295    cg12445832  0      +
    # 1      834356      834356    cg23999112  0      +
    # 1      837536      837536    cg11527153  0      -
    # 1      838379      838379    cg27573606  0      +
    # 1      838486      838486    cg04195702  0      -

    ng = 0 ## number of genes/loci processed
    nskip = 0 ## number of loci with missing data
    for line in bedFP:
        if DEBUG: print("[readBed6AnnotatioFileToDict][DEBUG] line:", line)
        dataA = line.strip('\n').split('\t')
        if dataA[0] == "chrom":
            continue    ## skip header line

        my_name = dataA[3]
        my_lociH[my_name]  = {}

        if dataA[0] == "NA":
            nskip = nskip +1
            continue

        my_lociH[my_name]["chrom"]      = str(dataA[0]) # {1:22, X, Y}
        my_lociH[my_name]["chromStart"] = int(dataA[1])
        my_lociH[my_name]["chromEnd"]   = int(dataA[2])
        my_lociH[my_name]["strand"]     = str(dataA[5])


        if DEBUG:
            if my_name == "cg13191808": 
                print("[readBed6AnnotatioFileToDict][DEBUG] cg13191808:", my_lociH[my_name])

        ng = ng + 1

    bedFP.close()

    print("[readBed6AnnotatioFileToDict][INFO] Skipped (NA)", nskip, "loci from", my_bed6AnnotFile)
    print("[readBed6AnnotatioFileToDict][INFO] Processed", len(my_lociH), "loci from", my_bed6AnnotFile)
    return(my_lociH)

#### Read through file and report on p-values #######################################
def reportPvalues(my_ecpgDataFile):
    ecpgdfFP = open(my_ecpgDataFile, "r")

    pvalsA = []
    nskip = 0 ## number of mappings with missing data
    for line in ecpgdfFP:
        if DEBUG: print("[reportPvalues][DEBUG] line:", line)
        dataA = line.strip('\n').split(',')
        if dataA[0] == "gt_id":
            continue    ## skip header line
        
        if dataA[5] == "":
            nskip = nskip +1
            continue

        mt_p   = float(dataA[5])
        pvalsA.append(mt_p)

    pvalsA = np.array(pvalsA)
    print("[reportPvalues][INFO] P-values skipped (missing data):", nskip)
    print("[reportPvalues][INFO] P-values read:",  len(pvalsA))
    print("[reportPvalues][INFO] P < 0.000001",    len(pvalsA[pvalsA < 0.000001]))
    print("[reportPvalues][INFO] P < 0.0000001",   len(pvalsA[pvalsA < 0.0000001]))
    print("[reportPvalues][INFO] P < 0.00000001",  len(pvalsA[pvalsA < 0.00000001]))
    print("[reportPvalues][INFO] P < 0.000000001", len(pvalsA[pvalsA < 0.000000001]))

    #sys.exit(55)
    return

#### Assign region for each eCpG #######################################
## TODO:
##  - support non-default p-value cut-off 
def assignRegion(my_ecpgDataFile, gH, mH):
    my_eqtmA = []

    my_typeCountH = {}
    my_typeCountH["trans"] = 0
    my_typeCountH["distal"] = 0
    my_typeCountH["cis"] = 0
    my_typeCountH["promoter"] = 0
    my_typeCountH["genebody"] = 0

    ecpgdfFP = open(my_ecpgDataFile, "r")

    # gt_id         mt_id       mt_est       mt_err       mt_t       mt_p
    # ILMN_2383229  cg13191808  1.3042431    0.24092653   5.4134474  5.9604645e-08
    # ILMN_2383229  cg17276863  -0.87402654  0.17654708   -4.950671  7.1525574e-07
    # ILMN_2383229  cg22478121  1.2773991    0.2258715    5.6554236  0.0
    # ILMN_2383229  cg08473330  0.7276063    0.14499585   5.018118   5.364418e-07
    # ILMN_2383229  cg02583418  0.41245365   0.080135465  5.1469555  2.3841858e-07
    # ILMN_2383229  cg04789550  1.6001295    0.2965917    5.395058   5.9604645e-08
    # ILMN_2383229  cg23039279  1.1443882    0.23120244   4.9497237  7.1525574e-07
    # ILMN_1806310  cg24348240  0.7328489    0.1438987    5.092811   3.5762787e-07
    # ILMN_1806310  cg05045517  0.92643046   0.14619319   6.337029   0.0

    nlp = 0 ## number loci processed
    ne = 0 ## number loci excluded
    npvalx = 0
    negx = 0
    nemt = 0
    npskip = 0

    for line in ecpgdfFP:
        if DEBUG: print("[assignRegion][DEBUG] line:", line)
        dataA = line.strip('\n').split(',')
        if dataA[0] == "gt_id":
            continue    ## skip header line

        if dataA[5]  == "":
            print("[assignRegion][INFO] P-value missing. Excluding loci", gt_id, mt_id, mt_p)
            npskip = npskip +1
            continue

        gt_id  = str(dataA[0])
        mt_id  = str(dataA[1])
        mt_est = float(dataA[2])
        mt_err = float(dataA[3])
        mt_t   = float(dataA[4])
        mt_p   = np.float32(dataA[5])

        nlp = nlp + 1

        #if mt_p > PVALCUTOFF:
        #print("[assignRegion][DEBUG] p-value:", nlp, mt_p)
        if mt_p > 0.000001:
            print("[assignRegion][INFO] P-value too large. Excluding loci", gt_id, mt_id, mt_p)
            ne = ne +1
            npvalx = npvalx +1
            continue

        try:
            mH[mt_id]
        except:
            print("[assignRegion][INFO] Annotation missing - methylation:", nlp, mt_id)
            ne = ne +1
            nemt = nemt +1
            continue

        try:
            gH[gt_id]
        except:
            print("[assignRegion][INFO] Annotation missing - gene expression:", nlp, gt_id)
            ne = ne +1
            negx = negx +1
            continue

        if DEBUG: print("[assignRegion][DEBUG]",nlp,"mt:", mt_id, mH[mt_id]["chrom"], "gx:", gt_id, gH[gt_id]["chrom"])

        ## cpgA
        ## mt_id mt_chrom mt_chromStart mt_strand gt_id gt_chrom gt_chromStart gt_strand region

        ##
        ## check for TRANS
        ##

        ## chr_mt != chr_gene

        try:
            mH[mt_id]["chrom"]
        except:
            print("[assignRegion][INFO] Annotation missing - methylation [chrom]:", nlp, mt_id)
            ne = ne +1
            nemt = nemt +1
            continue

        try:
            gH[gt_id]["chrom"]
        except:
            print("[assignRegion][INFO] Annotation missing - gene expression [chrom]:", nlp, gt_id)
            ne = ne +1
            negx = negx +1
            continue

        if mH[mt_id]["chrom"] != gH[gt_id]["chrom"]:
            cpgA = []
            cpgA.append(mt_id)
            cpgA.append(mH[mt_id]["chrom"])
            cpgA.append(mH[mt_id]["chromStart"])
            cpgA.append(mH[mt_id]["strand"])
            cpgA.append(gt_id)
            cpgA.append(gH[gt_id]["chrom"])
            cpgA.append(gH[gt_id]["chromStart"])
            cpgA.append(gH[gt_id]["strand"])
            cpgA.append("TRANS")
            my_typeCountH["trans"] += 1

            if DEBUG: print("[assignRegion][DEBUG]",nlp,cpgA)
            my_eqtmA.append(cpgA)

            if DEBUG: 
                if my_typeCountH["trans"] < 5:
                    print("[assignRegion][DEBUG]",nlp, len(my_eqtmA), my_eqtmA)

            #if DEBUG:
            #    if nlp >= 10:
            #        print("[assignRegion][WARN] member 8:", my_eqtmA[9]) ## print one from the list
            #        print("[assignRegion][WARN] Ending loop early:", nlp, " Are you debugging?")
            #    break

            continue   ## Move on to the next. CpGs cannot be TRANS && another region

        ##
        ## check for DISTAL - positive strand
        ##

        # DISTAL: > 50Kb upstream from TSS 

        #                         |>>>>>>>| gene
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        #                         | TSS (strand=“+”)
        #               | -offset (50Kb)
        #               | region start
        #       | cpg
        # no upstream window limit
        # ...-----------| target region
        # upstream                       downstream

        if gH[gt_id]["strand"] == "+":
            cpg_pos = mH[mt_id]["chromStart"] 
            geneStart_pos = gH[gt_id]["chromStart"]
            regionRef_pos = geneStart_pos - DISTAL_OFFSET
            if (cpg_pos < regionRef_pos):
                cpgA = []
                cpgA.append(mt_id)
                cpgA.append(mH[mt_id]["chrom"])
                cpgA.append(mH[mt_id]["chromStart"])
                cpgA.append(mH[mt_id]["strand"])
                cpgA.append(gt_id)
                cpgA.append(gH[gt_id]["chrom"])
                cpgA.append(gH[gt_id]["chromStart"])
                cpgA.append(gH[gt_id]["strand"])
                cpgA.append("DISTAL")
                my_typeCountH["distal"] += 1
                if DEBUG: print("[assignRegion][DEBUG]",nlp,cpgA)
                my_eqtmA.append(cpgA)
                #continue   ## allow eCpG to be annotated for multiple regions on the same chromosome


        ##
        ## check for CIS - positive strand
        ##

        # CIS:    < 50Kb upstream from TSS

        #                         |>>>>>>>| gene
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        #                         | TSS (strand=“+”)
        #                         | -offset (0Kb)
        #                 | -region start (50Kb)
        #                    | cpg
        #                 |-------| target region
        # upstream                                   downstream


        if gH[gt_id]["strand"] == "+":
            cpg_pos = mH[mt_id]["chromStart"] 
            geneStart_pos = gH[gt_id]["chromStart"]
            regionRef_pos = geneStart_pos - CIS_OFFSET
            regionUpStreamRange = geneStart_pos - CIS_UPSTREAM_DISTANCE
            if (regionUpStreamRange < cpg_pos) & (cpg_pos < geneStart_pos):
                cpgA = []
                cpgA.append(mt_id)
                cpgA.append(mH[mt_id]["chrom"])
                cpgA.append(mH[mt_id]["chromStart"])
                cpgA.append(mH[mt_id]["strand"])
                cpgA.append(gt_id)
                cpgA.append(gH[gt_id]["chrom"])
                cpgA.append(gH[gt_id]["chromStart"])
                cpgA.append(gH[gt_id]["strand"])
                cpgA.append("CIS")
                my_typeCountH["cis"] += 1
                if DEBUG: print("[assignRegion][DEBUG]",nlp,cpgA)
                my_eqtmA.append(cpgA)
                #continue   ## allow eCpG to be annotated for multiple regions on the same chromosome


        ##
        ## check for PROMOTER - positive strand
        ##

        # PROMOTER: +/- 2500bp from TSS

        #                         |>>>>>>>| gene
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        #                         | TSS (strand=“+”)
        #                         | -offset (0Kb)
        #                 | region start (-2500)
        #                      | cpg
        #                                 | region end (+2500)
        #                 |---------------| target region
        # upstream                                   downstream

        if gH[gt_id]["strand"] == "+":
            cpg_pos = mH[mt_id]["chromStart"] 
            geneStart_pos = gH[gt_id]["chromStart"]
            regionRef_pos = geneStart_pos - PROMOTER_OFFSET
            regionUpStreamRange = regionRef_pos - PROMOTER_UPSTREAM_DISTANCE
            regionDnStreamRange = regionRef_pos + PROMOTER_DOWNSTREAM_DISTANCE
            if (regionUpStreamRange < cpg_pos) & (cpg_pos < regionDnStreamRange):
                cpgA = []
                cpgA.append(mt_id)
                cpgA.append(mH[mt_id]["chrom"])
                cpgA.append(mH[mt_id]["chromStart"])
                cpgA.append(mH[mt_id]["strand"])
                cpgA.append(gt_id)
                cpgA.append(gH[gt_id]["chrom"])
                cpgA.append(gH[gt_id]["chromStart"])
                cpgA.append(gH[gt_id]["strand"])
                cpgA.append("PROMOTER")
                my_typeCountH["promoter"] += 1
                if DEBUG: print("[assignRegion][DEBUG]",nlp,cpgA)
                my_eqtmA.append(cpgA)
                #continue   ## allow eCpG to be annotated for multiple regions on the same chromosome

        ##
        ## check for GENE BODY - positive strand
        ##

        # GENE BODY: TSS to TES

        #                         |>>>>>>>| gene
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        #                         | TSS (strand=“+”)
        #                         | -offset (0Kb)
        #                         | region start (-2500)
        #                             | cpg
        #                                 | TES
        #                                 | region end (+2500)
        #                         |-------| target region
        # upstream                                   downstream

        if gH[gt_id]["strand"] == "+":
            cpg_pos = mH[mt_id]["chromStart"] 
            geneStart_pos = gH[gt_id]["chromStart"]
            geneEnd_pos = gH[gt_id]["chromEnd"]
            if (geneStart_pos < cpg_pos) & (cpg_pos < geneEnd_pos):
                cpgA = []
                cpgA.append(mt_id)
                cpgA.append(mH[mt_id]["chrom"])
                cpgA.append(mH[mt_id]["chromStart"])
                cpgA.append(mH[mt_id]["strand"])
                cpgA.append(gt_id)
                cpgA.append(gH[gt_id]["chrom"])
                cpgA.append(gH[gt_id]["chromStart"])
                cpgA.append(gH[gt_id]["strand"])
                cpgA.append("GENEBODY")
                my_typeCountH["genebody"] += 1
                if DEBUG: print("[assignRegion][DEBUG]",nlp,cpgA)
                my_eqtmA.append(cpgA)
                #continue   ## allow eCpG to be annotated for multiple regions on the same chromosome

        ##
        ## check for DISTAL - negative strand
        ##

        # DISTAL: > 50Kb upstream from TSS

        #       |<<<<<<<| gene
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        #               | TSS (strand=“-”)
        #                       | +offset (50Kb)
        #                       | region start
        #                               | cpg
        #                               no upstream window limit
        #                       |--------------... target region
        # downstream                                   upstream

        if gH[gt_id]["strand"] == "-":
            cpg_pos = mH[mt_id]["chromStart"] 
            geneStart_pos = gH[gt_id]["chromStart"]
            regionRef_pos = geneStart_pos + DISTAL_OFFSET
            if (regionRef_pos < cpg_pos):
                cpgA = []
                cpgA.append(mt_id)
                cpgA.append(mH[mt_id]["chrom"])
                cpgA.append(mH[mt_id]["chromStart"])
                cpgA.append(mH[mt_id]["strand"])
                cpgA.append(gt_id)
                cpgA.append(gH[gt_id]["chrom"])
                cpgA.append(gH[gt_id]["chromStart"])
                cpgA.append(gH[gt_id]["strand"])
                cpgA.append("DISTAL")
                my_typeCountH["distal"] += 1
                if DEBUG: print("[assignRegion][DEBUG]",nlp,cpgA)
                my_eqtmA.append(cpgA)
                #continue   ## allow eCpG to be annotated for multiple regions on the same chromosome

        ##
        ## check for CIS - negative strand
        ##

        # CIS:    < 50Kb upstream from TSS

        #           |<<<<<<| gene
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        #                  | TSS (strand=“-”)
        #                         | +offset (50Kb)
        #                         | region end 
        #                      | cpg
        #                  |------| target region
        # downstream                                   upstream

        if gH[gt_id]["strand"] == "-":
            cpg_pos = mH[mt_id]["chromStart"] 
            geneStart_pos = gH[gt_id]["chromStart"]
            regionRef_pos = geneStart_pos
            regionUpStreamRange = regionRef_pos + CIS_OFFSET
            if (geneStart_pos < cpg_pos ) & (cpg_pos < regionUpStreamRange):
                cpgA = []
                cpgA.append(mt_id)
                cpgA.append(mH[mt_id]["chrom"])
                cpgA.append(mH[mt_id]["chromStart"])
                cpgA.append(mH[mt_id]["strand"])
                cpgA.append(gt_id)
                cpgA.append(gH[gt_id]["chrom"])
                cpgA.append(gH[gt_id]["chromStart"])
                cpgA.append(gH[gt_id]["strand"])
                cpgA.append("CIS")
                my_typeCountH["cis"] += 1
                if DEBUG: print("[assignRegion][DEBUG]",nlp,cpgA)
                my_eqtmA.append(cpgA)
                #continue   ## allow eCpG to be annotated for multiple regions on the same chromosome

        ##
        ## check for PROMOTER - negative strand
        ##

        # PROMOTER: +/- 2500bp from TSS
    
        #           |<<<<<<| gene
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        #                  | TSS (strand=“-”)
        #                  | +offset (0Kb)
        #              | region start (-2500)
        #                      | cpg
        #                             | region end (+2500)
        #              |--------------| target region
        # downstream                                   upstream

        if gH[gt_id]["strand"] == "-":
            cpg_pos = mH[mt_id]["chromStart"] 
            geneStart_pos = gH[gt_id]["chromStart"]
            regionRef_pos = geneStart_pos + PROMOTER_OFFSET
            regionDnStreamRange = regionRef_pos - PROMOTER_DOWNSTREAM_DISTANCE
            regionUpStreamRange = regionRef_pos + PROMOTER_UPSTREAM_DISTANCE
            if (regionDnStreamRange < cpg_pos) & (cpg_pos < regionUpStreamRange):
                cpgA = []
                cpgA.append(mt_id)
                cpgA.append(mH[mt_id]["chrom"])
                cpgA.append(mH[mt_id]["chromStart"])
                cpgA.append(mH[mt_id]["strand"])
                cpgA.append(gt_id)
                cpgA.append(gH[gt_id]["chrom"])
                cpgA.append(gH[gt_id]["chromStart"])
                cpgA.append(gH[gt_id]["strand"])
                cpgA.append("PROMOTER")
                my_typeCountH["promoter"] += 1
                if DEBUG: print("[assignRegion][DEBUG]",nlp,cpgA)
                my_eqtmA.append(cpgA)
                #continue   ## allow eCpG to be annotated for multiple regions on the same chromosome

        ##
        ## check for GENE BODY - negative strand
        ##

        # GENE BODY: TSS to TES

        #                         |<<<<<<<| gene
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        #                         | TSS (strand=“-”)
        #                         | -offset (0Kb)
        #                         | region start (-2500)
        #                             | cpg
        #                                 | TES
        #                                 | region end (+2500)
        #                         |-------| target region
        # downstream                                   upstream

        if gH[gt_id]["strand"] == "-":
            cpg_pos = mH[mt_id]["chromStart"] 
            geneStart_pos = gH[gt_id]["chromStart"]
            geneEnd_pos = gH[gt_id]["chromEnd"]
            if (geneEnd_pos < cpg_pos) & (cpg_pos < geneStart_pos):
                cpgA = []
                cpgA.append(mt_id)
                cpgA.append(mH[mt_id]["chrom"])
                cpgA.append(mH[mt_id]["chromStart"])
                cpgA.append(mH[mt_id]["strand"])
                cpgA.append(gt_id)
                cpgA.append(gH[gt_id]["chrom"])
                cpgA.append(gH[gt_id]["chromStart"])
                cpgA.append(gH[gt_id]["strand"])
                cpgA.append("GENEBODY")
                my_typeCountH["genebody"] += 1
                if DEBUG: print("[assignRegion][DEBUG]",nlp,cpgA)
                my_eqtmA.append(cpgA)
                #continue   ## allow eCpG to be annotated for multiple regions on the same chromosome

        #if nlp >= 100:
        #    print("[assignRegion][WARN] Ending loop early:", nlp, " Are you debugging?")
        #    break

    print("[assignRegion][INFO] eCpgs Processed:", nlp, "Assigned:", len(my_eqtmA), "Excluded (any):", ne)
    print("[assignRegion][INFO] eCpgs Excluded:", "p-value filter:", npvalx, "p-value missing:", npskip, "gx annotation:", negx, "mt annotation:", nemt)
    print("[assignRegion][INFO] eCpgs Counts by Region:", my_typeCountH)
    return(my_eqtmA)

#### Give some usage information for this script #######################################
def usage(errorNum):
    print()
    print("assignRegionToEcpg.py - assign a region class to eCpGs")
    print()
    print("usage: assignRegionToEcpg.py [hD] -d <tecpg eQTM output> -g <gene annotation file> -m <methylation annotation file> -o <outfile name>")
    print(" e.g.: assignRegionToEcpg.py -d 1-1.csv -g G.bed6 -m M.bed6 -o ecpg.annot.csv")
    print()

    sys.exit(errorNum)

#### main #######################################
def main(argv):
    ecpgDataFile = ""
    geneAnnotFile = ""
    outFileName = ""
    eqtmH = {}

    try:
        opts, args = getopt.getopt(argv, "hd:g:m:o:D", ["help", "ecpgDataFile", "geneAnnotFile", "methylAnnotFile", "outFileName", "debug"])
    except getopt.GetoptError as err:
        print("[MAIN][ERROR] Unexpected options detected:",err)
        usage(20)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage(21)
        if opt in ("-d", "--ecpgDataFile"):
            ecpgDataFile = arg
        if opt in ("-g", "--geneAnnotFile"):
            geneAnnotFile = arg
        if opt in ("-m", "--methylAnnotFile"):
            methylAnnotFile = arg
        if opt in ("-o", "--outFileName"):
            outFileName = arg
        elif opt in ("-D", "--debug"):
            global DEBUG
            print("[MAIN][DEBUG] DEBUG set.")
            DEBUG = 1

    if ecpgDataFile == "":
        print("[MAIN][ERROR] eCpG data file missing.")
        usage(202)
    else:
        print("[MAIN][INFO] eCpG datafile:", ecpgDataFile)

    if geneAnnotFile == "":
        print("[MAIN][ERROR] gene annotation file missing.")
        usage(203)
    else:
        print("[MAIN][INFO] gene anntoation file:", geneAnnotFile)

    if methylAnnotFile == "":
        print("[MAIN][ERROR] methylation annotation file missing.")
        usage(204)
    else:
        print("[MAIN][INFO] methylation anntoation file:", methylAnnotFile)

    if outFileName == "":
        print("[MAIN][ERROR] output file name missing.")
        usage(205)
    else:
        print("[MAIN][INFO] output file name:", outFileName)

    ## Read in the gene annotation file to a dictionary
    geneH = readBed6AnnotatioFileToDict(geneAnnotFile)

    ## Read in the methylation annotation file to a dictionary
    methylH = readBed6AnnotatioFileToDict(methylAnnotFile)

    ## summarize the p-values
    print("[MAIN][INFO] Using default p-value cutoff of", PVALCUTOFF)
    reportPvalues(ecpgDataFile)

    ## Annotate the pairs
    eqtmA = assignRegion(ecpgDataFile, geneH, methylH)

    ## export the data

    ## eqtmA
    ## mt_id mt_chrom mt_chromStart mt_strand gt_id gt_chrom gt_chromStart gt_strand region

    print("[MAIN][INFO] Saving annotated data to:",outFileName)
    #regionData.to_csv(outFileName, sep=',', float_format='%.4f', index_label="region", compression='gzip')
    outfileFP = open(outFileName, "w")

    headerA=["mt_id","mt_chrom","mt_chromStart","mt_strand","gt_id","gt_chrom","gt_chromStart","gt_strand","region"]
    if DEBUG: print("[MAIN][DEBUG] headerA:", headerA)
    headerS = ",".join(headerA)
    outfileFP.write(headerS + "\n")
    n = 0
    for ecpg in eqtmA:
        lineA = []
        for i in ecpg:
            lineA.append(str(i))
        lineS = ",".join(lineA)
        outfileFP.write(lineS+"\n")
        #if n > 10:
        #    break
        #n += 1
    outfileFP.close()

#### Start here. #######################################
if __name__ == "__main__":
    main(sys.argv[1:])
